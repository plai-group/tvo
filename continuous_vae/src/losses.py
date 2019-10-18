import torch
from src import util
import numpy as np


def get_log_weight_log_p_log_q(generative_model, inference_network, obs, num_particles=1, reparam=True):
    """Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size, obs_dim]
        num_particles: int

    Returns:
        log_weight: tensor of shape [batch_size, num_particles]
        log_p: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
    """
    latent_dist = inference_network.get_latent_dist(obs)
    latent = inference_network.sample_from_latent_dist(latent_dist, num_particles, reparam=reparam)

    log_p = generative_model.get_log_prob(latent, obs).transpose(0, 1)
    log_q = inference_network.get_log_prob_from_latent_dist(latent_dist, latent).transpose(0, 1)
    log_weight = log_p - log_q
    return log_weight, log_p, log_q


def get_log_weight_and_log_q(generative_model, inference_network, obs, num_particles=1):
    """Compute log weight and log prob of inference network.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        log_weight: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
    """
    log_weight, log_p, log_q = get_log_weight_log_p_log_q(
        generative_model, inference_network, obs, num_particles=num_particles)
    return log_weight, log_q


def get_test_log_evidence(generative_model, inference_network, obs, num_particles=10):
    with torch.no_grad():
        log_weight, log_q = get_log_weight_and_log_q(
            generative_model, inference_network, obs, num_particles)
        log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
        iwae_log_evidence = torch.mean(log_evidence)
    return iwae_log_evidence


def get_iwae_loss(generative_model, inference_network, obs, args, valid_size):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """

    log_weight, log_q = get_log_weight_and_log_q(generative_model, inference_network, obs, args.S)

    stable_log_weight = log_weight - torch.max(log_weight, 1)[0].unsqueeze(1)
    weight = torch.exp(stable_log_weight)
    normalized_weight = weight / torch.sum(weight, 1).unsqueeze(1)

    loss = -torch.mean(torch.sum(normalized_weight.detach() * log_weight, 1), 0)

    iwae_estimate = get_test_log_evidence(generative_model, inference_network, obs, valid_size)

    return loss, iwae_estimate


def get_vae_loss(generative_model, inference_network, obs, args, valid_size):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """

    log_weight, log_q = get_log_weight_and_log_q(generative_model, inference_network, obs, args.S)
    train_elbo = torch.mean(log_weight)
    loss = -train_elbo

    iwae_estimate = get_test_log_evidence(generative_model, inference_network, obs, valid_size)

    return loss, iwae_estimate


def get_reinforce_loss(generative_model, inference_network, obs, args, valid_size):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_q = get_log_weight_and_log_q(generative_model, inference_network, obs, args.S)
    reinforce = log_weight.detach() * log_q + log_weight
    loss = -torch.mean(reinforce)

    iwae_estimate = get_test_log_evidence(generative_model, inference_network, obs, valid_size)

    return loss, iwae_estimate


def get_thermo_loss(generative_model, inference_network, obs, args, valid_size):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        partition: partition of [0, 1];
            tensor of shape [num_partitions + 1] where partition[0] is zero and
            partition[-1] is one;
            see https://en.wikipedia.org/wiki/Partition_of_an_interval
        num_particles: int
        integration: left, right or trapz

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_p, log_q = get_log_weight_log_p_log_q(
        generative_model, inference_network, obs, num_particles=args.S, reparam=False)
    thermo_loss = get_thermo_loss_from_log_weight_log_p_log_q(
        log_weight, log_p, log_q, args.partition, num_particles=args.S, integration=args.integration)
    iwae_estimate = get_test_log_evidence(generative_model, inference_network, obs, valid_size)

    return thermo_loss, iwae_estimate


def get_thermo_loss_from_log_weight_log_p_log_q(log_weight, log_p, log_q, partition, num_particles=1,
                                                integration='left'):
    """Args:
        log_weight: tensor of shape [batch_size, num_particles]
        log_p: tensor of shape [batch_size, num_particles]
        log_q: tensor of shape [batch_size, num_particles]
        partition: partition of [0, 1];
            tensor of shape [num_partitions + 1] where partition[0] is zero and
            partition[-1] is one;
            see https://en.wikipedia.org/wiki/Partition_of_an_interval
        num_particles: int
        integration: left, right or trapz

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """


    heated_log_weight = log_weight.unsqueeze(-1) * partition
    heated_normalized_weight = util.exponentiate_and_normalize(
        heated_log_weight, dim=1)
    thermo_logp = partition * log_p.unsqueeze(-1) + \
        (1 - partition) * log_q.unsqueeze(-1)

    wf = heated_normalized_weight * log_weight.unsqueeze(-1)
    w_detached = heated_normalized_weight.detach()
    wf_detached = wf.detach()
    if num_particles == 1:
        correction = 1
    else:
        correction = num_particles / (num_particles - 1)

    cov = correction * torch.sum(
        w_detached * (log_weight.unsqueeze(-1) - torch.sum(wf, dim=1, keepdim=True)).detach() *
        (thermo_logp - torch.sum(thermo_logp * w_detached, dim=1, keepdim=True)),
        dim=1)

    multiplier = torch.zeros_like(partition)
    if integration == 'trapz':
        multiplier[0] = 0.5 * (partition[1] - partition[0])
        multiplier[1:-1] = 0.5 * (partition[2:] - partition[0:-2])
        multiplier[-1] = 0.5 * (partition[-1] - partition[-2])
    elif integration == 'left':
        multiplier[:-1] = partition[1:] - partition[:-1]
    elif integration == 'right':
        multiplier[1:] = partition[1:] - partition[:-1]

    loss = -torch.mean(torch.sum(
        multiplier * (cov + torch.sum(
            w_detached * log_weight.unsqueeze(-1), dim=1)),
        dim=1))

    return loss


def get_log_p_and_kl(generative_model, inference_network, obs, num_samples):
    """Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_samples: int

    Returns:
        log_p: tensor of shape [batch_size]
        kl: tensor of shape [batch_size]
    """

    log_weight, _ = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_samples)
    log_p = torch.logsumexp(log_weight, dim=1) - np.log(num_samples)
    elbo = torch.mean(log_weight, dim=1)
    kl = log_p - elbo
    return log_p, kl
