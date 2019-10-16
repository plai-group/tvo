import torch
import util
import numpy as np
import train
import data


def run(args):
    # set up args
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        args.cuda = True
    else:
        device = torch.device('cpu')
        args.cuda = False
    if args.train_mode == 'thermo' or args.train_mode == 'thermo_wake':
        partition = util.get_partition(
            args.num_partitions, args.partition_type, args.log_beta_min,
            device)
    util.print_with_time('device = {}'.format(device))
    util.print_with_time(str(args))

    # save args
    save_dir = util.get_save_dir()
    args_path = util.get_args_path(save_dir)
    util.save_object(args, args_path)

    # data
    binarized_mnist_train, binarized_mnist_valid, binarized_mnist_test = \
        data.load_binarized_mnist(where=args.where)
    data_loader = data.get_data_loader(
        binarized_mnist_train, args.batch_size, device)
    valid_data_loader = data.get_data_loader(
        binarized_mnist_valid, args.valid_batch_size, device)
    test_data_loader = data.get_data_loader(
        binarized_mnist_test, args.test_batch_size, device)
    train_obs_mean = torch.tensor(np.mean(binarized_mnist_train, axis=0),
                                  device=device, dtype=torch.float)

    # init models
    util.set_seed(args.seed)
    generative_model, inference_network = util.init_models(
        train_obs_mean, args.architecture, device)

    # optim
    optim_kwargs = {'lr': args.learning_rate}

    # train
    if args.train_mode == 'ws':
        train_callback = train.TrainWakeSleepCallback(
            save_dir, args.num_particles * args.batch_size, test_data_loader,
            args.eval_num_particles, args.logging_interval,
            args.checkpoint_interval, args.eval_interval)
        train.train_wake_sleep(
            generative_model, inference_network, data_loader,
            args.num_iterations, args.num_particles, optim_kwargs,
            train_callback)
    elif args.train_mode == 'ww':
        train_callback = train.TrainWakeWakeCallback(
            save_dir, args.num_particles, test_data_loader,
            args.eval_num_particles, args.logging_interval,
            args.checkpoint_interval, args.eval_interval)
        train.train_wake_wake(generative_model, inference_network, data_loader,
                              args.num_iterations, args.num_particles,
                              optim_kwargs, train_callback)
    elif args.train_mode == 'reinforce' or args.train_mode == 'vimco':
        train_callback = train.TrainIwaeCallback(
            save_dir, args.num_particles, args.train_mode, test_data_loader,
            args.eval_num_particles, args.logging_interval,
            args.checkpoint_interval, args.eval_interval)
        train.train_iwae(args.train_mode, generative_model, inference_network,
                         data_loader, args.num_iterations, args.num_particles,
                         optim_kwargs, train_callback)
    elif args.train_mode == 'thermo':
        train_callback = train.TrainThermoCallback(
            save_dir, args.num_particles, partition, test_data_loader,
            args.eval_num_particles, args.logging_interval,
            args.checkpoint_interval, args.eval_interval)
        train.train_thermo(generative_model, inference_network, data_loader,
                           args.num_iterations, args.num_particles, partition,
                           optim_kwargs, train_callback)
    elif args.train_mode == 'thermo_wake':
        train_callback = train.TrainThermoWakeCallback(
            save_dir, args.num_particles, test_data_loader,
            args.eval_num_particles, args.logging_interval,
            args.checkpoint_interval, args.eval_interval)
        train.train_thermo_wake(generative_model, inference_network,
                                data_loader, args.num_iterations,
                                args.num_particles, partition, optim_kwargs,
                                train_callback)

    # eval validation
    train_callback.valid_log_p, train_callback.valid_kl = train.eval_gen_inf(
        generative_model, inference_network, valid_data_loader,
        args.eval_num_particles)

    # save models and stats
    util.save_checkpoint(save_dir, iteration=None,
                         generative_model=generative_model,
                         inference_network=inference_network)
    stats_path = util.get_stats_path(save_dir)
    util.save_object(train_callback, stats_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-mode', default='ww',
                        help='ww, ws, reinforce, vimco, thermo, thermo_wake')
    parser.add_argument('--architecture', default='linear_1',
                        help='linear_1, linear_2, linear_3 or non_linear')
    parser.add_argument('--batch-size', type=int, default=24,
                        help=' ')
    parser.add_argument('--eval-num-particles', type=int, default=5,
                        help=' ')
    parser.add_argument('--valid-batch-size', type=int, default=100,
                        help=' ')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help=' ')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help=' ')
    parser.add_argument('--num-iterations', type=int, default=100000,
                        help=' ')
    parser.add_argument('--logging-interval', type=int, default=1000,
                        help=' ')
    parser.add_argument('--eval-interval', type=int, default=1000,
                        help=' ')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help=' ')
    parser.add_argument('--num-particles', type=int, default=2,
                        help=' ')
    parser.add_argument('--num-partitions', type=int, default=10,
                        help='only used in training with thermo objective; '
                             'corresponds to number of betas')
    parser.add_argument('--log-beta-min', type=float, default=-10,
                        help='log base ten of beta_min')
    parser.add_argument('--partition-type', default='log',
                        help='log or linear')
    parser.add_argument('--where', default='local',
                        help='local or cc_cedar')
    parser.add_argument('--seed', type=int, default=1, help=' ')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    run(args)
