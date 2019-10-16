import torch
import numpy as np
import util
import time
import data
import train


def main():
    util.print_with_time('torch.__version__ = {}'.format(torch.__version__))
    num_runs = 10
    num_particles_list = [2, 5, 10, 50, 100, 500, 1000, 5000]
    num_partitions_list = [2, 5, 10, 50, 100, 500, 1000]
    memory_thermo = np.zeros(
        (len(num_particles_list), len(num_partitions_list), num_runs))
    time_thermo = np.zeros(
        (len(num_particles_list), len(num_partitions_list), num_runs))
    memory_vimco = np.zeros((len(num_particles_list), num_runs))
    time_vimco = np.zeros((len(num_particles_list), num_runs))
    memory_reinforce = np.zeros((len(num_particles_list), num_runs))
    time_reinforce = np.zeros((len(num_particles_list), num_runs))
    num_iterations = 100

    device = torch.device('cuda')
    generative_model, inference_network = util.init_models(
        None, 'linear_2', device)
    batch_size = 24
    binarized_mnist_dir = data.BINARIZED_MNIST_DIR_CC
    binarized_mnist_train, binarized_mnist_valid, _ = \
        data.load_binarized_mnist(dir=binarized_mnist_dir)
    data_loader = data.get_data_loader(
        binarized_mnist_train, batch_size, device)
    optim_kwargs = {}
    for i, num_particles in enumerate(num_particles_list):
        for j, num_partitions in enumerate(num_partitions_list):
            partition = util.get_partition(
                num_partitions, 'log', device=device)
            for k in range(num_runs):
                torch.cuda.reset_max_memory_allocated(device=device)
                start = time.time()
                train.train_thermo(generative_model, inference_network,
                                   data_loader, num_iterations,
                                   num_particles, partition, optim_kwargs)
                end = time.time()
                memory_thermo[i, j, k] = \
                    torch.cuda.max_memory_allocated(device=device)
                time_thermo[i, j, k] = end - start
                print('thermo {} {} {} memory = {}MB, time = {}s'.format(
                    num_particles, num_partitions, k,
                    memory_thermo[i, j, k] / 1e6, time_thermo[i, j, k]))

    for i, num_particles in enumerate(num_particles_list):
        for k in range(num_runs):
            torch.cuda.reset_max_memory_allocated(device=device)
            start = time.time()
            train.train_iwae('vimco', generative_model, inference_network,
                             data_loader, num_iterations, num_particles,
                             optim_kwargs)
            end = time.time()
            memory_vimco[i, k] = torch.cuda.max_memory_allocated(device=device)
            time_vimco[i, k] = end - start
            util.print_with_time(
                'vimco {} {} memory = {}MB, time = {}s'.format(
                    num_particles, k, memory_vimco[i, k] / 1e6,
                    time_vimco[i, k]))

            torch.cuda.reset_max_memory_allocated(device=device)
            start = time.time()
            train.train_iwae('reinforce', generative_model, inference_network,
                             data_loader, num_iterations, num_particles,
                             optim_kwargs)
            end = time.time()
            memory_reinforce[i, k] = torch.cuda.max_memory_allocated(
                device=device)
            time_reinforce[i, k] = end - start
            util.print_with_time(
                'reinforce {} {} memory = {}MB, time = {}s'.format(
                    num_particles, k, memory_reinforce[i, k] / 1e6,
                    time_reinforce[i, k]))

    path = './save/efficiency.pkl'
    util.save_object((memory_thermo, time_thermo, memory_vimco, time_vimco,
                      memory_reinforce, time_reinforce), path)


if __name__ == '__main__':
    main()
