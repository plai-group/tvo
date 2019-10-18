import itertools
import pickle
from pathlib import Path
from types import SimpleNamespace
import h5py

import numpy as np
import torch

from src import losses
from src import util
from src.data import get_data_loader

from sacred import Experiment
from sacred.observers import TinyDbObserver
from sacred.observers import FileStorageObserver
ex = Experiment()


@ex.config
def my_config():
    # paths
    data_dir     = './data/'
    model_dir    = './models'
    data_name    = 'data.pkl'

    # Model
    architecture = 'non_linear'
    loss         = 'vae'
    hidden_dim   = 200
    latent_dim   = 50
    integration  = 'left'
    model_type   = 'continuous'
    cuda         = True
    num_stochastic_layers    = 1
    num_deterministic_layers = 2
    num_deterministic_layers = 0 if architecture == 'linear' else num_deterministic_layers

    # Hyper
    K  = 5
    S  = 10
    lr = 0.001
    test_K = 20
    log_beta_min   = -1.09
    partition_type = "log"

    # Training
    seed       = 1
    epochs     = 5000
    batch_size = 1000
    valid_S    = 10
    test_S     = 5000
    test_batch_size = 1

    optimizer  = "adam"
    checkpoint_frequency  = int(epochs / 5)
    checkpoint = False
    checkpoint = checkpoint if checkpoint_frequency > 0 else False

    test_frequency = 5
    test_during_training = True
    train_only = False

    save_grads = True

    # Assertions
    assert partition_type in ['log', 'linear']
    assert architecture in ["linear", "non_linear"]
    assert integration in ['left', 'right', 'trap']
    assert model_type in ['continuous', 'discrete']
    assert optimizer in ['adam']
    assert loss in ['reinforce', 'vae', 'iwae', 'thermo']

    # Name
    model_name = "{}.{}.{}.{}.{}".format(architecture, loss, num_stochastic_layers, num_deterministic_layers, S)
    if loss == 'thermo':
        model_name = "{}.{}.{}.{}".format(model_name, integration, log_beta_min, K)


def init(config, _run):
    args = SimpleNamespace(**config)
    util.seed_all(args.seed)

    args._run = _run

    _id = str(_run._id) if _run._id else 'local'
    args.model_name = str(_id) + "_" + _run.experiment_info["name"]

    losses_dict = {
        'vae': losses.get_vae_loss,
        'reinforce': losses.get_reinforce_loss,
        'thermo': losses.get_thermo_loss,
        'iwae': losses.get_iwae_loss
    }

    args.loss_name = args.loss
    args.loss = losses_dict[args.loss]

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False

    args.data_path = Path(args.data_dir) / args.data_name
    return args


@ex.capture
def log_scalar(name, scalar, step=None, _run=None, verbose=True):
    if isinstance(scalar, torch.Tensor):
        scalar = scalar.item()

    if step is not None:
        if verbose: print("Epoch: {} - {}: {}".format(step, name, scalar))
        _run.log_scalar(name, scalar, step)
    else:
        if verbose: print("{}: {}".format(name, scalar))
        _run.log_scalar(name, scalar)


@ex.capture
def save_checkpoint(generative_model, inference_network, epoch, loss, optimizer, args, _run=None, _config=None):
    path = Path(args.model_dir) / (args.model_name + "." + str(args.loss_name) + ".epoch." + str(epoch) + ".model")
    print("Saving checkpoint: {}".format(path))

    torch.save({'epoch': epoch,
                'generative_model_state_dict': generative_model.state_dict(),
                'inference_network_state_dict': inference_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'last_loss': loss,
                'config': dict(_config)}, path)

    _run.add_artifact(path)


def train(args):
    # read data
    with args._run.open_resource(args.data_path, 'rb') as file_handle:
        data = pickle.load(file_handle)

    train_image = data['train_image']
    test_image  = data['test_image']

    train_data_loader = get_data_loader(train_image, args.batch_size, args)
    test_data_loader  = get_data_loader(test_image, args.test_batch_size, args)

    # Make models
    train_obs_mean = util.tensor(np.mean(train_image, axis=0), args)
    generative_model, inference_network = util.init_models(train_obs_mean, args)

    # Make partition
    args.partition = util.get_partition(args.K, args.partition_type, args.log_beta_min, args.device)

    # Make optimizer
    parameters = itertools.chain.from_iterable([x.parameters() for x in [generative_model, inference_network]])
    optimizer = torch.optim.Adam(parameters, lr=args.lr)


    for epoch in range(args.epochs):
        epoch_train_elbo = 0
        for idx, data in enumerate(train_data_loader):
            optimizer.zero_grad()
            loss, elbo = args.loss(generative_model, inference_network, data, args, args.valid_S)
            loss.backward()
            optimizer.step()
            epoch_train_elbo += elbo.item()

        if (args.save_grads and (epoch % args.test_frequency) == 0):
            # Save grads
            grad_variance = util.calculate_grad_variance(generative_model, inference_network, data, args)
            log_scalar("grad.variance", grad_variance, epoch, verbose=True)

        if torch.isnan(loss):
            break

        epoch_train_elbo = epoch_train_elbo / len(train_data_loader)
        log_scalar("train.elbo", epoch_train_elbo, epoch)


        if (args.checkpoint and (epoch != 0) and ((epoch % args.checkpoint_frequency) == 0)):
            save_checkpoint(generative_model, inference_network, epoch, epoch_train_elbo, optimizer, args)

        if args.train_only: continue

        # run test set
        if (epoch == (args.epochs - 1)) or \
           (args.test_during_training and ((epoch % args.test_frequency) == 0)):
            print("Running test set...")
            test_elbo = 0
            with torch.no_grad():
                for idx, data in enumerate(test_data_loader):
                    _, elbo = args.loss(generative_model, inference_network, data, args, args.test_S)
                    test_elbo += elbo.item()

            test_elbo = test_elbo / len(test_data_loader)
            log_scalar("test.elbo", test_elbo, epoch)

        # ------ end of training loop ---------

    # Save trained model
    if args.checkpoint:
        save_checkpoint(generative_model, inference_network, epoch, epoch_train_elbo, optimizer, args)

    if args.train_only:
        return None
    else:
        results = {
            "test_elbo": test_elbo if not args.train_only else None
            }
        return results


@ex.automain
def experiment(_config, _run):
    args = init(_config, _run)
    result = train(args)
    return result

