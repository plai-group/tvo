import torch
import losses
import util
import itertools


def eval_gen_inf(generative_model, inference_network, data_loader,
                 num_particles):
    log_p_total = 0
    kl_total = 0
    num_data = 0
    for obs in iter(data_loader):
        log_p, kl = losses.get_log_p_and_kl(
            generative_model, inference_network, obs,
            num_particles)
        log_p_total += torch.sum(log_p).item()
        kl_total += torch.sum(kl).item()
        num_data += obs.shape[0]
    return log_p_total / num_data, kl_total / num_data


def train_wake_sleep(generative_model, inference_network, data_loader,
                     num_iterations, num_particles, optim_kwargs,
                     callback=None):
    optimizer_phi = torch.optim.Adam(inference_network.parameters(),
                                     **optim_kwargs)
    optimizer_theta = torch.optim.Adam(generative_model.parameters(),
                                       **optim_kwargs)

    iteration = 0
    while iteration < num_iterations:
        for obs in iter(data_loader):
            # wake theta
            optimizer_phi.zero_grad()
            optimizer_theta.zero_grad()
            wake_theta_loss, elbo = losses.get_wake_theta_loss(
                generative_model, inference_network, obs, num_particles)
            wake_theta_loss.backward()
            optimizer_theta.step()

            # sleep phi
            optimizer_phi.zero_grad()
            optimizer_theta.zero_grad()
            sleep_phi_loss = losses.get_sleep_loss(
                generative_model, inference_network,
                num_samples=obs.shape[0] * num_particles)
            sleep_phi_loss.backward()
            optimizer_phi.step()

            if callback is not None:
                callback(iteration, wake_theta_loss.item(),
                         sleep_phi_loss.item(), elbo.item(), generative_model,
                         inference_network, optimizer_theta, optimizer_phi)

            iteration += 1
            # by this time, we have gone through `iteration` iterations
            if iteration == num_iterations:
                break


class DontPickleCuda:
    # https://stackoverflow.com/questions/2345944/exclude-objects-field-from-pickling-in-python
    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle test_data_loader
        if 'test_data_loader' in state:
            del state['test_data_loader']
        if 'partition' in state:
            del state['partition']
        if 'test_obs' in state:
            del state['test_obs']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class TrainWakeSleepCallback(DontPickleCuda):
    def __init__(self, save_dir, num_samples, test_data_loader,
                 eval_num_particles=5000, logging_interval=10,
                 checkpoint_interval=100, eval_interval=10):
        self.save_dir = save_dir
        self.num_samples = num_samples
        self.test_data_loader = test_data_loader
        self.eval_num_particles = eval_num_particles
        self.logging_interval = logging_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.wake_theta_loss_history = []
        self.sleep_phi_loss_history = []
        self.elbo_history = []
        self.log_p_history = []
        self.kl_history = []
        self.grad_std_history = []

    def __call__(self, iteration, wake_theta_loss, sleep_phi_loss, elbo,
                 generative_model, inference_network, optimizer_theta,
                 optimizer_phi):
        if iteration % self.logging_interval == 0:
            util.print_with_time(
                'Iteration {} losses: theta = {:.3f}, phi = {:.3f}, elbo = '
                '{:.3f}'.format(iteration, wake_theta_loss, sleep_phi_loss,
                                elbo))
            self.wake_theta_loss_history.append(wake_theta_loss)
            self.sleep_phi_loss_history.append(sleep_phi_loss)
            self.elbo_history.append(elbo)

        if iteration % self.checkpoint_interval == 0:
            stats_path = util.get_stats_path(self.save_dir)
            util.save_object(self, stats_path)
            util.save_checkpoint(
                self.save_dir, iteration,
                generative_model=generative_model,
                inference_network=inference_network)

        if iteration % self.eval_interval == 0:
            log_p, kl = eval_gen_inf(
                generative_model, inference_network, self.test_data_loader,
                self.eval_num_particles)
            self.log_p_history.append(log_p)
            self.kl_history.append(kl)

            stats = util.OnlineMeanStd()
            for _ in range(10):
                inference_network.zero_grad()
                sleep_phi_loss = losses.get_sleep_loss(
                    generative_model, inference_network, self.num_samples)
                sleep_phi_loss.backward()
                stats.update([p.grad for p in inference_network.parameters()])
            self.grad_std_history.append(stats.avg_of_means_stds()[1].item())
            util.print_with_time(
                'Iteration {} log_p = {:.3f}, kl = {:.3f}'.format(
                    iteration, self.log_p_history[-1], self.kl_history[-1]))


def train_wake_wake(generative_model, inference_network, data_loader,
                    num_iterations, num_particles, optim_kwargs,
                    callback=None):
    optimizer_phi = torch.optim.Adam(inference_network.parameters(),
                                     **optim_kwargs)
    optimizer_theta = torch.optim.Adam(generative_model.parameters(),
                                       **optim_kwargs)

    iteration = 0
    while iteration < num_iterations:
        for obs in iter(data_loader):
            log_weight, log_q = losses.get_log_weight_and_log_q(
                generative_model, inference_network, obs, num_particles)

            # wake theta
            optimizer_phi.zero_grad()
            optimizer_theta.zero_grad()
            wake_theta_loss, elbo = losses.get_wake_theta_loss_from_log_weight(
                log_weight)
            wake_theta_loss.backward(retain_graph=True)
            optimizer_theta.step()

            # wake phi
            optimizer_phi.zero_grad()
            optimizer_theta.zero_grad()
            wake_phi_loss = losses.get_wake_phi_loss_from_log_weight_and_log_q(
                log_weight, log_q)
            wake_phi_loss.backward()
            optimizer_phi.step()

            if callback is not None:
                callback(iteration, wake_theta_loss.item(),
                         wake_phi_loss.item(), elbo.item(), generative_model,
                         inference_network, optimizer_theta, optimizer_phi)

            iteration += 1
            # by this time, we have gone through `iteration` iterations
            if iteration == num_iterations:
                break


class TrainWakeWakeCallback(DontPickleCuda):
    def __init__(self, save_dir, num_particles, test_data_loader,
                 eval_num_particles=5000, logging_interval=10,
                 checkpoint_interval=100, eval_interval=10):
        self.save_dir = save_dir
        self.num_particles = num_particles
        self.test_data_loader = test_data_loader
        self.eval_num_particles = eval_num_particles
        self.logging_interval = logging_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.wake_theta_loss_history = []
        self.wake_phi_loss_history = []
        self.elbo_history = []
        self.log_p_history = []
        self.kl_history = []
        self.grad_std_history = []
        self.test_obs = next(iter(test_data_loader))

    def __call__(self, iteration, wake_theta_loss, wake_phi_loss, elbo,
                 generative_model, inference_network, optimizer_theta,
                 optimizer_phi):
        if iteration % self.logging_interval == 0:
            util.print_with_time(
                'Iteration {} losses: theta = {:.3f}, phi = {:.3f}, elbo = '
                '{:.3f}'.format(iteration, wake_theta_loss, wake_phi_loss,
                                elbo))
            self.wake_theta_loss_history.append(wake_theta_loss)
            self.wake_phi_loss_history.append(wake_phi_loss)
            self.elbo_history.append(elbo)

        if iteration % self.checkpoint_interval == 0:
            stats_path = util.get_stats_path(self.save_dir)
            util.save_object(self, stats_path)
            util.save_checkpoint(
                self.save_dir, iteration,
                generative_model=generative_model,
                inference_network=inference_network)

        if iteration % self.eval_interval == 0:
            log_p, kl = eval_gen_inf(
                generative_model, inference_network, self.test_data_loader,
                self.eval_num_particles)
            self.log_p_history.append(log_p)
            self.kl_history.append(kl)

            stats = util.OnlineMeanStd()
            for _ in range(10):
                generative_model.zero_grad()
                wake_theta_loss, elbo = losses.get_wake_theta_loss(
                    generative_model, inference_network, self.test_obs,
                    self.num_particles)
                wake_theta_loss.backward()
                theta_grads = [p.grad.clone() for p in
                               generative_model.parameters()]

                inference_network.zero_grad()
                wake_phi_loss = losses.get_wake_phi_loss(
                    generative_model, inference_network, self.test_obs,
                    self.num_particles)
                wake_phi_loss.backward()
                phi_grads = [p.grad for p in inference_network.parameters()]

                stats.update(theta_grads + phi_grads)
            self.grad_std_history.append(stats.avg_of_means_stds()[1].item())
            util.print_with_time(
                'Iteration {} log_p = {:.3f}, kl = {:.3f}'.format(
                    iteration, self.log_p_history[-1], self.kl_history[-1]))


def train_iwae(algorithm, generative_model, inference_network, data_loader,
               num_iterations, num_particles, optim_kwargs, callback=None):
    parameters = itertools.chain.from_iterable(
        [x.parameters() for x in [generative_model, inference_network]])
    optimizer = torch.optim.Adam(parameters, **optim_kwargs)

    iteration = 0
    while iteration < num_iterations:
        for obs in iter(data_loader):
            optimizer.zero_grad()
            if algorithm == 'vimco':
                loss, elbo = losses.get_vimco_loss(
                    generative_model, inference_network, obs, num_particles)
            elif algorithm == 'reinforce':
                loss, elbo = losses.get_reinforce_loss(
                    generative_model, inference_network, obs, num_particles)
            loss.backward()
            optimizer.step()

            if callback is not None:
                callback(iteration, loss.item(), elbo.item(), generative_model,
                         inference_network, optimizer)

            iteration += 1
            # by this time, we have gone through `iteration` iterations
            if iteration == num_iterations:
                break


class TrainIwaeCallback(DontPickleCuda):
    def __init__(self, save_dir, num_particles, train_mode, test_data_loader,
                 eval_num_particles=5000, logging_interval=10,
                 checkpoint_interval=100, eval_interval=10):
        self.save_dir = save_dir
        self.num_particles = num_particles
        self.test_data_loader = test_data_loader
        self.eval_num_particles = eval_num_particles
        self.logging_interval = logging_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.loss_history = []
        self.elbo_history = []
        self.log_p_history = []
        self.kl_history = []
        self.grad_std_history = []
        self.test_obs = next(iter(test_data_loader))
        self.train_mode = train_mode

    def __call__(self, iteration, loss, elbo, generative_model,
                 inference_network, optimizer):
        if iteration % self.logging_interval == 0:
            util.print_with_time(
                'Iteration {} loss = {:.3f}, elbo = {:.3f}'.format(
                    iteration, loss, elbo))
            self.loss_history.append(loss)
            self.elbo_history.append(elbo)

        if iteration % self.checkpoint_interval == 0:
            stats_path = util.get_stats_path(self.save_dir)
            util.save_object(self, stats_path)
            util.save_checkpoint(
                self.save_dir, iteration,
                generative_model=generative_model,
                inference_network=inference_network)

        if iteration % self.eval_interval == 0:
            log_p, kl = eval_gen_inf(
                generative_model, inference_network, self.test_data_loader,
                self.eval_num_particles)
            self.log_p_history.append(log_p)
            self.kl_history.append(kl)

            stats = util.OnlineMeanStd()
            for _ in range(10):
                generative_model.zero_grad()
                inference_network.zero_grad()
                if self.train_mode == 'vimco':
                    loss, elbo = losses.get_vimco_loss(
                        generative_model, inference_network, self.test_obs,
                        self.num_particles)
                elif self.train_mode == 'reinforce':
                    loss, elbo = losses.get_reinforce_loss(
                        generative_model, inference_network, self.test_obs,
                        self.num_particles)
                loss.backward()
                stats.update([p.grad for p in generative_model.parameters()] +
                             [p.grad for p in inference_network.parameters()])
            self.grad_std_history.append(stats.avg_of_means_stds()[1].item())
            util.print_with_time(
                'Iteration {} log_p = {:.3f}, kl = {:.3f}'.format(
                    iteration, self.log_p_history[-1], self.kl_history[-1]))


def train_thermo(generative_model, inference_network, data_loader,
                 num_iterations, num_particles, partition, optim_kwargs,
                 callback=None):
    parameters = itertools.chain.from_iterable(
        [x.parameters() for x in [generative_model, inference_network]])
    optimizer = torch.optim.Adam(parameters, **optim_kwargs)

    iteration = 0
    while iteration < num_iterations:
        for obs in iter(data_loader):
            optimizer.zero_grad()
            loss, elbo = losses.get_thermo_loss(
                generative_model, inference_network, obs, partition,
                num_particles)
            loss.backward()
            optimizer.step()

            if callback is not None:
                callback(iteration, loss.item(), elbo.item(), generative_model,
                         inference_network, optimizer)

            iteration += 1
            # by this time, we have gone through `iteration` iterations
            if iteration == num_iterations:
                break


class TrainThermoCallback(DontPickleCuda):
    def __init__(self, save_dir, num_particles, partition, test_data_loader,
                 eval_num_particles=5000, logging_interval=10,
                 checkpoint_interval=100, eval_interval=10):
        self.save_dir = save_dir
        self.num_particles = num_particles
        self.test_data_loader = test_data_loader
        self.eval_num_particles = eval_num_particles
        self.logging_interval = logging_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.loss_history = []
        self.elbo_history = []
        self.log_p_history = []
        self.kl_history = []
        self.grad_std_history = []
        self.test_obs = next(iter(test_data_loader))
        self.partition = partition

    def __call__(self, iteration, loss, elbo, generative_model,
                 inference_network, optimizer):
        if iteration % self.logging_interval == 0:
            util.print_with_time(
                'Iteration {} loss = {:.3f}, elbo = {:.3f}'.format(
                    iteration, loss, elbo))
            self.loss_history.append(loss)
            self.elbo_history.append(elbo)

        if iteration % self.checkpoint_interval == 0:
            stats_path = util.get_stats_path(self.save_dir)
            util.save_object(self, stats_path)
            util.save_checkpoint(
                self.save_dir, iteration,
                generative_model=generative_model,
                inference_network=inference_network)

        if iteration % self.eval_interval == 0:
            log_p, kl = eval_gen_inf(
                generative_model, inference_network, self.test_data_loader,
                self.eval_num_particles)
            self.log_p_history.append(log_p)
            self.kl_history.append(kl)

            stats = util.OnlineMeanStd()
            for _ in range(10):
                generative_model.zero_grad()
                inference_network.zero_grad()
                loss, elbo = losses.get_thermo_loss(
                    generative_model, inference_network, self.test_obs,
                    self.partition, self.num_particles)
                loss.backward()
                stats.update([p.grad for p in generative_model.parameters()] +
                             [p.grad for p in inference_network.parameters()])
            self.grad_std_history.append(stats.avg_of_means_stds()[1].item())
            util.print_with_time(
                'Iteration {} log_p = {:.3f}, kl = {:.3f}'.format(
                    iteration, self.log_p_history[-1], self.kl_history[-1]))


def train_thermo_wake(generative_model, inference_network, data_loader,
                      num_iterations, num_particles, partition, optim_kwargs,
                      callback=None):
    optimizer_phi = torch.optim.Adam(inference_network.parameters(),
                                     **optim_kwargs)
    optimizer_theta = torch.optim.Adam(generative_model.parameters(),
                                       **optim_kwargs)

    iteration = 0
    while iteration < num_iterations:
        for obs in iter(data_loader):
            log_weight, log_p, log_q = losses.get_log_weight_log_p_log_q(
                generative_model, inference_network, obs, num_particles)

            # wake theta
            optimizer_phi.zero_grad()
            optimizer_theta.zero_grad()
            thermo_loss, elbo = \
                losses.get_thermo_loss_from_log_weight_log_p_log_q(
                    log_weight, log_p, log_q, partition, num_particles)
            thermo_loss.backward(retain_graph=True)
            optimizer_theta.step()

            # wake phi
            optimizer_phi.zero_grad()
            optimizer_theta.zero_grad()
            wake_phi_loss = losses.get_wake_phi_loss_from_log_weight_and_log_q(
                log_weight, log_q)
            wake_phi_loss.backward()
            optimizer_phi.step()

            if callback is not None:
                callback(iteration, thermo_loss.item(),
                         wake_phi_loss.item(), elbo.item(), generative_model,
                         inference_network, optimizer_theta, optimizer_phi)

            iteration += 1
            # by this time, we have gone through `iteration` iterations
            if iteration == num_iterations:
                break


class TrainThermoWakeCallback(DontPickleCuda):
    def __init__(self, save_dir, num_particles, test_data_loader,
                 eval_num_particles=5000, logging_interval=10,
                 checkpoint_interval=100, eval_interval=10):
        self.save_dir = save_dir
        self.num_particles = num_particles
        self.test_data_loader = test_data_loader
        self.eval_num_particles = eval_num_particles
        self.logging_interval = logging_interval
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.thermo_loss_history = []
        self.wake_phi_loss_history = []
        self.elbo_history = []
        self.log_p_history = []
        self.kl_history = []
        self.grad_std_history = []
        self.test_obs = next(iter(test_data_loader))

    def __call__(self, iteration, thermo_loss, wake_phi_loss, elbo,
                 generative_model, inference_network, optimizer_theta,
                 optimizer_phi):
        if iteration % self.logging_interval == 0:
            util.print_with_time(
                'Iteration {} losses: theta = {:.3f}, phi = {:.3f}, elbo = '
                '{:.3f}'.format(iteration, thermo_loss, wake_phi_loss,
                                elbo))
            self.thermo_loss_history.append(thermo_loss)
            self.wake_phi_loss_history.append(wake_phi_loss)
            self.elbo_history.append(elbo)

        if iteration % self.checkpoint_interval == 0:
            stats_path = util.get_stats_path(self.save_dir)
            util.save_object(self, stats_path)
            util.save_checkpoint(
                self.save_dir, iteration,
                generative_model=generative_model,
                inference_network=inference_network)

        if iteration % self.eval_interval == 0:
            log_p, kl = eval_gen_inf(
                generative_model, inference_network, self.test_data_loader,
                self.eval_num_particles)
            self.log_p_history.append(log_p)
            self.kl_history.append(kl)

            stats = util.OnlineMeanStd()
            for _ in range(10):
                inference_network.zero_grad()
                wake_phi_loss = losses.get_wake_phi_loss(
                    generative_model, inference_network, self.test_obs,
                    self.num_particles)
                wake_phi_loss.backward()
                stats.update([p.grad for p in inference_network.parameters()])
            self.grad_std_history.append(stats.avg_of_means_stds()[1].item())
            util.print_with_time(
                'Iteration {} log_p = {:.3f}, kl = {:.3f}'.format(
                    iteration, self.log_p_history[-1], self.kl_history[-1]))
