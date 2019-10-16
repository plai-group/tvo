import torch
import util
import numpy as np


def get_sleep_loss(generative_model, inference_network, num_samples=1):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """

    latent, obs = generative_model.sample_latent_and_obs(num_samples)
    return -torch.mean(inference_network.get_log_prob(latent, obs))


def get_log_weight_log_p_log_q(generative_model, inference_network, obs,
                               num_particles=1):
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
    latent = inference_network.sample_from_latent_dist(latent_dist,
                                                       num_particles)
    log_p = generative_model.get_log_prob(latent, obs).transpose(0, 1)
    log_q = inference_network.get_log_prob_from_latent_dist(
        latent_dist, latent).transpose(0, 1)
    log_weight = log_p - log_q
    return log_weight, log_p, log_q


def get_log_weight_and_log_q(generative_model, inference_network, obs,
                             num_particles=1):
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


def get_wake_theta_loss_from_log_weight(log_weight):
    """Args:
        log_weight: tensor of shape [batch_size, num_particles]

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """

    _, num_particles = log_weight.shape
    elbo = torch.mean(
        torch.logsumexp(log_weight, dim=1) - np.log(num_particles))
    return -elbo, elbo


def get_wake_theta_loss(generative_model, inference_network, obs,
                        num_particles=1):
    """Scalar that we call .backward() on and step the optimizer.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, _ = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)
    return get_wake_theta_loss_from_log_weight(log_weight)


def get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q):
    """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    normalized_weight = util.exponentiate_and_normalize(log_weight, dim=1)
    return torch.mean(-torch.sum(normalized_weight.detach() * log_q, dim=1))


def get_wake_phi_loss(generative_model, inference_network, obs,
                      num_particles=1):
    """
    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)
    return get_wake_phi_loss_from_log_weight_and_log_q(log_weight, log_q)


def get_reinforce_loss(generative_model, inference_network, obs,
                       num_particles=1):
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
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)

    # this is term 1 in equation (2) of https://arxiv.org/pdf/1805.10469.pdf
    reinforce_correction = log_evidence.detach() * torch.sum(log_q, dim=1)

    elbo = torch.mean(log_evidence)
    loss = -elbo - torch.mean(reinforce_correction)
    return loss, elbo


def get_vimco_loss_slow(generative_model, inference_network, obs,
                        num_particles=1):
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
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)
    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
    reinforce_correction = 0
    for i in range(num_particles):
        log_weight_ = log_weight[:, util.range_except(num_particles, i)]

        # this is the B term in VIMCO gradient in
        # https://arxiv.org/pdf/1805.10469.pdf
        control_variate = torch.logsumexp(
            torch.cat([log_weight_,
                       torch.mean(log_weight_, dim=1, keepdim=True)], dim=1),
            dim=1) - np.log(num_particles)
        reinforce_correction = reinforce_correction + \
            (log_evidence.detach() - control_variate.detach()) * log_q[:, i]

    elbo = torch.mean(log_evidence)
    loss = -elbo - torch.mean(reinforce_correction)
    return loss, elbo


def get_vimco_loss(generative_model, inference_network, obs, num_particles=1):
    """Almost twice faster version of VIMCO loss (measured for batch_size = 24,
        num_particles = 1000). Inspired by Adam Kosiorek's implementation.

    Args:
        generative_model: models.GenerativeModel object
        inference_network: models.InferenceNetwork object
        obs: tensor of shape [batch_size]
        num_particles: int

    Returns:

        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """
    log_weight, log_q = get_log_weight_and_log_q(
        generative_model, inference_network, obs, num_particles)

    # shape [batch_size, num_particles]
    # log_weight_[b, k] = 1 / (K - 1) \sum_{\ell \neq k} \log w_{b, \ell}
    log_weight_ = (torch.sum(log_weight, dim=1, keepdim=True) - log_weight) \
        / (num_particles - 1)

    # shape [batch_size, num_particles, num_particles]
    # temp[b, k, k_] =
    #     log_weight_[b, k]     if k == k_
    #     log_weight[b, k]      otherwise
    temp = log_weight.unsqueeze(-1) + torch.diag_embed(
        log_weight_ - log_weight)

    # this is the \Upsilon_{-k} term below equation 3
    # shape [batch_size, num_particles]
    control_variate = torch.logsumexp(temp, dim=1) - np.log(num_particles)

    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
    elbo = torch.mean(log_evidence)
    loss = -elbo - torch.mean(torch.sum(
        (log_evidence.unsqueeze(-1) - control_variate).detach() * log_q, dim=1
    ))

    return loss, elbo


def get_thermo_loss(generative_model, inference_network, obs,
                    partition=None, num_particles=1, integration='left',
                    mode='covariance'):
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
        mode: covariance or baselined_reinforce

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        elbo: average elbo over data
    """

    log_weight, log_p, log_q = get_log_weight_log_p_log_q(
        generative_model, inference_network, obs, num_particles=num_particles)

    return get_thermo_loss_from_log_weight_log_p_log_q(
        log_weight, log_p, log_q, partition, num_particles=num_particles,
        integration=integration, mode=mode)


def get_thermo_loss_from_log_weight_log_p_log_q(
    log_weight, log_p, log_q, partition, num_particles=1, integration='left',
    mode='covariance'
):
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
        mode: covariance or baselined_reinforce

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
    # wf_detached = wf.detach()
    if num_particles == 1:
        correction = 1
    else:
        correction = num_particles / (num_particles - 1)

    if mode == 'covariance':
        thing_to_add = correction * torch.sum(
            w_detached *
            (log_weight.unsqueeze(-1) -
             torch.sum(wf, dim=1, keepdim=True)).detach() *
            (thermo_logp -
             torch.sum(thermo_logp * w_detached, dim=1, keepdim=True)),
            dim=1)
    elif mode == 'baselined_reinforce':
        thing_to_add = correction * torch.sum(
            w_detached *
            (log_weight.unsqueeze(-1) -
             torch.sum(wf, dim=1, keepdim=True)).detach() *
            thermo_logp,
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
        multiplier * (thing_to_add + torch.sum(
            w_detached * log_weight.unsqueeze(-1), dim=1)),
        dim=1))

    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
    elbo = torch.mean(log_evidence)

    return loss, elbo


def get_thermo_loss_different_samples(
    generative_model, inference_network, obs, partition=None,
    num_particles=1, integration='left'
):
    """Thermo loss gradient estimator computed using two set of importance
        samples.

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

    log_weights, log_ps, log_qs, heated_normalized_weights = [], [], [], []
    for _ in range(2):
        log_weight, log_p, log_q = get_log_weight_log_p_log_q(
            generative_model, inference_network, obs, num_particles)
        log_weights.append(log_weight)
        log_ps.append(log_p)
        log_qs.append(log_q)

        heated_log_weight = log_weight.unsqueeze(-1) * partition
        heated_normalized_weights.append(util.exponentiate_and_normalize(
            heated_log_weight, dim=1))

    w_detached = heated_normalized_weights[0].detach()
    thermo_logp = partition * log_ps[0].unsqueeze(-1) + \
        (1 - partition) * log_qs[0].unsqueeze(-1)
    wf = heated_normalized_weights[1] * log_weights[1].unsqueeze(-1)

    if num_particles == 1:
        correction = 1
    else:
        correction = num_particles / (num_particles - 1)

    thing_to_add = correction * torch.sum(
        w_detached *
        (log_weight.unsqueeze(-1) -
         torch.sum(wf, dim=1, keepdim=True)).detach() *
        thermo_logp,
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
        multiplier * (thing_to_add + torch.sum(
            w_detached * log_weight.unsqueeze(-1), dim=1)),
        dim=1))

    log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(num_particles)
    elbo = torch.mean(log_evidence)

    return loss, elbo


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
