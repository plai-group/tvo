import torch
import numpy as np
import random

class AverageMeter(object):
    """
    Computes and stores the average, var, and sample_var
    Taken from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.M2   = 0

        self.mean = 0
        self.variance = 0
        self.sample_variance = 0

    def update(self, val):
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.M2 += delta * delta2

        self.variance = self.M2 / self.count if self.count > 2 else 0
        self.sample_variance = self.M2 / (self.count - 1)  if self.count > 2 else 0

def get_grads(model):
    return torch.cat([torch.flatten(p.grad.clone()) for p in model.parameters()]).cpu()


def tensor(data, args):
    return torch.tensor(np.array(data), device=args.device, dtype=torch.float)


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


def seed_all(seed):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



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
            partition[1:] = torch.logspace(log_beta_min, 0, steps=num_partitions, device=device,
                dtype=torch.float)
    return partition


def init_models(train_obs_mean, args):
    """Args:
        train_obs_mean: tensor of shape [obs_dim]
        architecture: linear_1, linear_2, linear_3 or non_linear
        device: torch.device

    Returns: generative_model, inference_network
    """

    if args.model_type == 'discrete':
        from src import discrete_models as models
    else:
        from src import continuous_models as models

    if args.architecture == 'linear':
        num_deterministic_layers = 0
    else:
        num_deterministic_layers = args.num_deterministic_layers

    generative_model = models.GenerativeModel(
        num_stochastic_layers=args.num_stochastic_layers,
        num_deterministic_layers=num_deterministic_layers,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        device=args.device,
        train_obs_mean=train_obs_mean)
    inference_network = models.InferenceNetwork(
        num_stochastic_layers=args.num_stochastic_layers,
        num_deterministic_layers=num_deterministic_layers,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        device=args.device,
        train_obs_mean=train_obs_mean)

    if args.device.type == 'cuda':
        generative_model.cuda()
        inference_network.cuda()

    return generative_model, inference_network


def calculate_grad_variance(generative_model, inference_network, data, args):
    grad_var   = AverageMeter()

    for _ in range(10):
        inference_network.zero_grad()
        generative_model.zero_grad()
        loss, elbo = args.loss(generative_model, inference_network, data, args, args.valid_S)
        loss.backward()
        model_grads, network_grads = get_grads(generative_model), get_grads(inference_network)
        grad_var.update(torch.cat((model_grads, network_grads)))

    grad_std = grad_var.variance.sqrt().mean()
    return grad_std


def get_reconstruction(generative_model, inference_network, obs, num_samples):
    """Args:
        generative_model:
        inference_network:
        obs: [num_test_obs, 784]
        num_samples: int

    Returns:
        obs_reconstruction: [num_test_obs, num_samples, 784]"""

    num_test_obs = obs.shape[0]
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

    def rsample(self, sample_shape=torch.Size()):
        sample_chain = self.chain_dist.rsample(sample_shape=sample_shape)
        sample_next = self.get_next_dist(sample_chain[-1]).rsample(
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

    def rsample(self, sample_shape=torch.Size()):
        return (self.single_dist.rsample(sample_shape=sample_shape),)

    def log_prob(self, value):
        return self.single_dist.log_prob(value[0])


class ReversedChainDistribution(torch.distributions.Distribution):
    def __init__(self, chain_dist):
        self.chain_dist = chain_dist

    def sample(self, sample_shape=torch.Size()):
        return tuple(reversed(self.chain_dist.sample(
            sample_shape=sample_shape)))

    def rsample(self, sample_shape=torch.Size()):
        return tuple(reversed(self.chain_dist.rsample(
            sample_shape=sample_shape)))

    def log_prob(self, value):
        return self.chain_dist.log_prob(tuple(reversed(value)))
