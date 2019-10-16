import datetime
import torch
import numpy as np
import os
import pickle
import uuid
import models
import data


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                                 exp(values[i_1, ..., i_N])
            log( ------------------------------------------------------------ )
                    sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    # log_numerator = values
    return values - log_denominator


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    return torch.exp(lognormexp(values, dim=dim))


def range_except(end, i):
    """Outputs an increasing list from 0 to (end - 1) except i.
    Args:
        end: int
        i: int

    Returns: list of length (end - 1)
    """

    result = list(set(range(end)))
    return result[:i] + result[(i + 1):]


def get_partition(num_partitions, partition_type, log_beta_min=-10,
                  device=None):
    """Create a non-decreasing sequence of values between zero and one.
    See https://en.wikipedia.org/wiki/Partition_of_an_interval.

    Args:
        num_partitions: length of sequence minus one
        partition_type: \'linear\' or \'log\'
        log_beta_min: log (base ten) of beta_min. only used if partition_type
            is log. default -10 (i.e. beta_min = 1e-10).
        device: torch.device object (cpu by default)

    Returns: tensor of shape [num_partitions + 1]
    """
    if device is None:
        device = torch.device('cpu')
    if num_partitions == 1:
        partition = torch.tensor([0, 1], dtype=torch.float, device=device)
    else:
        if partition_type == 'linear':
            partition = torch.linspace(0, 1, steps=num_partitions + 1,
                                       device=device)
        elif partition_type == 'log':
            partition = torch.zeros(num_partitions + 1, device=device,
                                    dtype=torch.float)
            partition[1:] = torch.logspace(
                log_beta_min, 0, steps=num_partitions, device=device,
                dtype=torch.float)
    return partition


def get_yyyymmdd():
    return str(datetime.date.today()).replace('-', '')


def get_hhmmss():
    return datetime.datetime.now().strftime('%H:%M:%S')


def print_with_time(str):
    print(get_yyyymmdd() + ' ' + get_hhmmss() + ' ' + str)


# https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
def save_object(obj, path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    print_with_time('Saved to {}'.format(path))


def load_object(path):
    with open(path, 'rb') as input_:
        obj = pickle.load(input_)
    return obj


def save_checkpoint(dir='.', iteration=None, **torch_objects):
    """Args:
        dir: directory where to save
        iteration: int
        **torch_objects: dict of models and optimizers to save
    """

    if iteration is None:
        suffix = ''
    else:
        suffix = iteration
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, 'checkpoint{}.pt'.format(suffix))
    torch.save({name: state_dict_to_cpu(torch_object.state_dict())
                for name, torch_object in torch_objects.items()}, path)
    print_with_time('Saved to {}'.format(path))


def load_checkpoint(dir='.', iteration=None, **torch_objects):
    """Args:
        dir: directory where to save
        iteration: int
        **torch_objects: dict of models and optimizers to load

    Returns: torch_objects: dict of models and optimizers with loaded params
    """
    if iteration is None:
        suffix = ''
    else:
        suffix = iteration
    path = os.path.join(dir, 'checkpoint{}.pt'.format(suffix))
    if os.path.exists(path):
        state_dicts = torch.load(path)
        result = {}
        for name, torch_object in torch_objects.items():
            result[name] = torch_object.load_state_dict(state_dicts[name])
        return result
    else:
        return None


def state_dict_to_cpu(state_dict):
    """Convert parameters in state_dict to cpu"""
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    return state_dict


def get_stats_path(dir='.'):
    return os.path.join(dir, 'stats.pkl')


def get_uuid():
    return str(uuid.uuid4())[:8]


def get_save_dir(root='./save/'):
    return os.path.join(root, get_yyyymmdd() + '_' + get_uuid())


class OnlineMeanStd():
    """Follows online algorithm from
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford
    """

    def __init__(self):
        self.count = 0
        self.means = None
        self.M2s = None

    def update(self, new_variables):
        if self.count == 0:
            self.count = 1
            self.means = []
            self.M2s = []
            for new_var in new_variables:
                self.means.append(new_var.data)
                self.M2s.append(new_var.data.new(new_var.size()).fill_(0))
        else:
            self.count = self.count + 1
            for new_var_idx, new_var in enumerate(new_variables):
                delta = new_var.data - self.means[new_var_idx]
                self.means[new_var_idx] = self.means[new_var_idx] + delta / \
                    self.count
                delta_2 = new_var.data - self.means[new_var_idx]
                self.M2s[new_var_idx] = self.M2s[new_var_idx] + delta * delta_2

    def means_stds(self):
        if self.count < 2:
            raise ArithmeticError('Need more than 1 value. Have {}'.format(
                self.count))
        else:
            stds = []
            for i in range(len(self.means)):
                stds.append(torch.sqrt(self.M2s[i] / self.count))
            return self.means, stds

    def avg_of_means_stds(self):
        means, stds = self.means_stds()
        num_parameters = np.sum([len(p) for p in means])
        return (np.sum([torch.sum(p) for p in means]) / num_parameters,
                np.sum([torch.sum(p) for p in stds]) / num_parameters)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_models(train_obs_mean, architecture, device):
    """Args:
        train_obs_mean: tensor of shape [obs_dim]
        architecture: linear_1, linear_2, linear_3 or non_linear
        device: torch.device

    Returns: generative_model, inference_network
    """

    if architecture[:len('linear')] == 'linear':
        num_stochastic_layers = int(architecture[-1])
        generative_model = models.GenerativeModel(
            num_stochastic_layers=num_stochastic_layers,
            num_deterministic_layers=0,
            device=device, train_obs_mean=train_obs_mean)
        inference_network = models.InferenceNetwork(
            num_stochastic_layers=num_stochastic_layers,
            num_deterministic_layers=0,
            device=device, train_obs_mean=train_obs_mean)
    elif architecture == 'non_linear':
        generative_model = models.GenerativeModel(
            num_stochastic_layers=1,
            num_deterministic_layers=2,
            device=device, train_obs_mean=train_obs_mean)
        inference_network = models.InferenceNetwork(
            num_stochastic_layers=1,
            num_deterministic_layers=2,
            device=device, train_obs_mean=train_obs_mean)

    if device.type == 'cuda':
        generative_model.cuda()
        inference_network.cuda()

    return generative_model, inference_network


def get_args_path(dir):
    return os.path.join(dir, 'args.pkl')


def args_match(dir, **kwargs):
    """Do training args match kwargs?"""

    args_path = get_args_path(dir)
    if os.path.exists(args_path):
        args = load_object(args_path)
        for k, v in kwargs.items():
            if k not in args.__dict__ or args.__dict__[k] != v:
                return False
        return True
    else:
        return False


def list_subdirs(root):
    for file in os.listdir(root):
        path = os.path.join(root, file)
        if os.path.isdir(path):
            yield(path)


def list_dirs_args_match(root, **kwargs):
    """Return a list of model folders whose training args
    match kwargs.
    """

    result = []
    for dir in list_subdirs(root):
        if args_match(dir, **kwargs):
            result.append(dir)
    return result


def get_most_recent_dir_args_match(root='./save/', **kwargs):
    dirs = list_dirs_args_match(root, **kwargs)
    if len(dirs) > 0:
        return dirs[np.argmax(
            [os.stat(x).st_mtime for x in dirs])]


def load_models(dir, iteration, where='local'):
    args = load_object(get_args_path(dir))
    binarized_mnist_train, _, _ = data.load_binarized_mnist(where=where)

    device = torch.device('cpu')
    train_obs_mean = torch.tensor(np.mean(binarized_mnist_train, axis=0),
                                  device=device, dtype=torch.float)
    generative_model, inference_network = init_models(
        train_obs_mean, args.architecture, device)
    load_checkpoint(dir, iteration, generative_model=generative_model,
                    inference_network=inference_network)
    return generative_model, inference_network


def get_reconstruction(generative_model, inference_network, obs, num_samples):
    """Args:
        generative_model:
        inference_network:
        obs: [num_test_obs, 784]
        num_samples: int

    Returns:
        obs_reconstruction: [num_test_obs, num_samples, 784]"""

    latent_dist = inference_network.get_latent_dist(obs)
    latent = inference_network.sample_from_latent_dist(
        latent_dist, num_samples)
    obs_dist = generative_model.get_obs_dist(latent)
    obs_reconstruction = obs_dist.sample()
    return torch.transpose(obs_reconstruction, 0, 1)


class ChainDistribution(torch.distributions.Distribution):
    def __init__(self, chain_dist, get_next_dist):
        self.chain_dist = chain_dist
        self.get_next_dist = get_next_dist

    def sample(self, sample_shape=torch.Size()):
        sample_chain = self.chain_dist.sample(sample_shape=sample_shape)
        sample_next = self.get_next_dist(sample_chain[-1]).sample(
            sample_shape=())
        return sample_chain + (sample_next,)

    def log_prob(self, value):
        log_prob_chain = self.chain_dist.log_prob(value[:-1])
        log_prob_next = self.get_next_dist(value[-2]).log_prob(value[-1])
        return log_prob_chain + log_prob_next


class ChainDistributionFromSingle(torch.distributions.Distribution):
    def __init__(self, single_dist):
        self.single_dist = single_dist

    def sample(self, sample_shape=torch.Size()):
        return (self.single_dist.sample(sample_shape=sample_shape),)

    def log_prob(self, value):
        return self.single_dist.log_prob(value[0])


class ReversedChainDistribution(torch.distributions.Distribution):
    def __init__(self, chain_dist):
        self.chain_dist = chain_dist

    def sample(self, sample_shape=torch.Size()):
        return tuple(reversed(self.chain_dist.sample(
            sample_shape=sample_shape)))

    def log_prob(self, value):
        return self.chain_dist.log_prob(tuple(reversed(value)))
