import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import util
import numpy as np
import os


def main(args):
    if args.mode == 'efficiency':
        num_runs = 10
        num_particles_list = [2, 5, 10, 50, 100, 500, 1000, 5000]
        num_partitions_list = [2, 5, 10, 50, 100, 500, 1000]
        path = './save/efficiency.pkl'
        (memory_thermo, time_thermo, memory_vimco, time_vimco,
         memory_reinforce, time_reinforce) = util.load_object(path)

        fig, axs = plt.subplots(1, 2, dpi=200, figsize=(6, 4))
        # colors = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8']
        norm = matplotlib.colors.Normalize(vmin=0,
                                           vmax=len(num_particles_list))
        cmap = matplotlib.cm.ScalarMappable(norm=norm,
                                            cmap=matplotlib.cm.Blues)
        cmap.set_array([])
        colors = [cmap.to_rgba(i + 1) for i in range(len(num_particles_list))]

        for i, num_partitions in enumerate(num_partitions_list):
            axs[0].plot(num_particles_list,
                        np.mean(time_thermo[:, i], axis=-1),
                        label='thermo K={}'.format(num_partitions),
                        color=colors[i], marker='x', linestyle='none')
        axs[0].plot(num_particles_list, np.mean(time_vimco, axis=-1),
                    color='black', label='vimco', marker='o', linestyle='none',
                    fillstyle='none')
        axs[0].plot(num_particles_list, np.mean(time_reinforce, axis=-1),
                    color='black', label='reinforce', marker='v',
                    linestyle='none', fillstyle='none')

        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlabel('number of particles')
        axs[0].set_ylabel('time (seconds)')
        axs[0].grid(True)
        axs[0].grid(True, which='minor', linewidth=0.2)
        # axs[0].legend(bbox_to_anchor=(1.13, -0.19), loc='upper center', ncol=3)
        sns.despine(ax=axs[0])

        # colors = ['C0', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8']
        for i, num_partitions in enumerate(num_partitions_list):
            axs[1].plot(num_particles_list,
                        np.mean(memory_thermo[:, i] / 1e6, axis=-1),
                        label='thermo K={}'.format(num_partitions),
                        color=colors[i], marker='x', linestyle='none')
        axs[1].plot(num_particles_list, np.mean(memory_vimco / 1e6, axis=-1),
                    color='black', label='vimco', marker='o', linestyle='none',
                    fillstyle='none')
        axs[1].plot(num_particles_list,
                    np.mean(memory_reinforce / 1e6, axis=-1),
                    color='black', label='reinforce', marker='v',
                    linestyle='none', fillstyle='none')

        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('number of particles')
        axs[1].set_ylabel('memory (MB)')
        axs[-1].legend(fontsize=6, ncol=2)
        axs[1].grid(True)
        axs[1].grid(True, which='minor', linewidth=0.2)
        sns.despine(ax=axs[1])

        fig.tight_layout()
        if not os.path.exists('./plots/'):
            os.makedirs('./plots/')
        filename = './plots/efficiency.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('saved to {}'.format(filename))
    elif args.mode == 'insights':
        markersize = 3
        learning_rate = 3e-4
        architecture = 'linear_3'
        seed = 8
        train_mode = 'thermo'
        num_particles_list = [2, 5, 10, 50]
        num_partitions_list = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        log_beta_mins_1 = [-10, -1, -0.045757490560675115]
        log_beta_mins_2 = [-5, -2, -1.6989700043360187, -1.5228787452803376,
                           -1.3979400086720375, -1.3010299956639813,
                           -1.2218487496163564, -1.1549019599857433,
                           -1.0969100130080565, -1.0457574905606752,
                           -1, -0.6989700043360187, -0.5228787452803375,
                           -0.3979400086720376, -0.3010299956639812,
                           -0.2218487496163564, -0.15490195998574313,
                           -0.09691001300805639, -0.045757490560675115]
        num_iterations = 400

        log_p_thermo_partition_sweep = np.full(
            (len(num_particles_list), len(log_beta_mins_1),
             len(num_partitions_list), num_iterations), np.nan)
        log_p_thermo_beta_sweep = np.full(
            (len(num_particles_list), len(log_beta_mins_2), num_iterations),
            np.nan)

        for num_particles_idx, num_particles in enumerate(num_particles_list):
            for log_beta_min_idx, log_beta_min in enumerate(log_beta_mins_1):
                for num_partitions_idx, num_partitions in enumerate(
                    num_partitions_list
                ):
                    dir_ = util.get_most_recent_dir_args_match(
                        train_mode=train_mode,
                        architecture=architecture,
                        learning_rate=learning_rate,
                        num_particles=num_particles,
                        num_partitions=num_partitions,
                        log_beta_min=log_beta_min,
                        seed=seed)
                    if dir_ is not None:
                        stats = util.load_object(util.get_stats_path(dir_))
                        log_p_thermo_partition_sweep[
                            num_particles_idx, log_beta_min_idx,
                            num_partitions_idx] = stats.log_p_history[:num_iterations]

                        print('thermo {} ({} partitions) beta_min = 1e{} after'
                              ' {} it: {}'.format(
                                num_particles, num_partitions, log_beta_min,
                                len(stats.log_p_history),
                                stats.log_p_history[-1]))
                    else:
                        print('missing')

            num_partitions = 2
            for log_beta_min_idx, log_beta_min in enumerate(log_beta_mins_2):
                dir_ = util.get_most_recent_dir_args_match(
                    train_mode=train_mode,
                    architecture=architecture,
                    learning_rate=learning_rate,
                    num_particles=num_particles,
                    num_partitions=num_partitions,
                    log_beta_min=log_beta_min,
                    seed=seed
                )
                if dir_ is not None:
                    stats = util.load_object(util.get_stats_path(dir_))
                    log_p_thermo_beta_sweep[
                        num_particles_idx,
                        log_beta_min_idx] = stats.log_p_history[:num_iterations]
                    print('thermo {} ({} partitions) beta_min = 1e{} after {}'
                          ' it: {}'.format(
                            num_particles, num_partitions, log_beta_min,
                            len(stats.log_p_history), stats.log_p_history[-1]))
                else:
                    print('missing')

        fig, axs = plt.subplots(2, 2, dpi=200,
                                figsize=(12, 7), sharey=True)


        for log_beta_min_idx, ax in zip(range(len(log_beta_mins_1)), [axs[0, 0], axs[0, 1], axs[1, 0]]):
            colors = ['C1', 'C2', 'C4', 'C5']
            # ax = axs[log_beta_min_idx]
            for num_particles_idx, num_particles in enumerate(
                num_particles_list
            ):
                ax.plot(
                    num_partitions_list,
                    log_p_thermo_partition_sweep[
                        num_particles_idx, log_beta_min_idx, :, -1],
                    color=colors[num_particles_idx],
                    label=num_particles,
                    marker='o',
                    markersize=markersize,
                    linestyle='solid',
                    linewidth=0.7)
            ax.set_title(r'$\beta_1 = {:.0e}$'.format(
                10**log_beta_mins_1[log_beta_min_idx]))
            # ax.set_xticks(np.arange(len(num_partitions_list)))
            # ax.set_xticklabels(num_partitions_list)
            ax.set_xlabel('number of partitions')
            ax.set_xticks(np.arange(0, max(num_partitions_list) + 1, 10))

        ax = axs[1, 1]
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            ax.plot(10**np.array(log_beta_mins_2),
                    log_p_thermo_beta_sweep[num_particles_idx, :, -1],
                    color=colors[num_particles_idx],
                    label=num_particles,
                    marker='o',
                    markersize=markersize,
                    linestyle='solid',
                    linewidth=0.7)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_title('2 partitions')
        ax.set_xlabel(r'$\beta_1$')

        print(np.max(log_p_thermo_beta_sweep[..., -1], axis=-1))
        print(np.argmax(log_p_thermo_beta_sweep[..., -1], axis=-1))
        print([log_beta_mins_2[i] for i in np.argmax(log_p_thermo_beta_sweep[..., -1], axis=-1)])
        print([10**log_beta_mins_2[i] for i in np.argmax(log_p_thermo_beta_sweep[..., -1], axis=-1)])
        # print(log_beta_mins_2[np.argmax(log_p_thermo_beta_sweep[..., -1], axis=-1)])

        for axx in axs:
            for ax in axx:
                ax.grid(True, axis='y')

        for ax in axs[:, 0]:
            ax.set_ylim(top=-88)
            ax.set_ylabel(r'$\log p(x)$')

        axs[1, 1].legend(title='number of particles', ncol=2,
                         loc='lower right')

        for axx in axs:
            for ax in axx:
                sns.despine(ax=ax, trim=True)

        # ax.('thermo')
        fig.tight_layout()
        if not os.path.exists('./plots/'):
            os.makedirs('./plots/')
        filename = './plots/insights.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('saved to {}'.format(filename))
    elif args.mode == 'baselines':
        learning_rate = 3e-4
        architecture = 'linear_3'
        seed = 8
        non_thermo_train_modes = ['ww', 'vimco']
        num_particles_list = [2, 5, 10, 50]
        num_partitions_list = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        # log_beta_mins_1 = [-10, -1, -0.045757490560675115]
        log_beta_mins_2 = [-5, -2, -1.6989700043360187, -1.5228787452803376,
                           -1.3979400086720375, -1.3010299956639813,
                           -1.2218487496163564, -1.1549019599857433,
                           -1.0969100130080565, -1.0457574905606752,
                           -1, -0.6989700043360187, -0.5228787452803375,
                           -0.3979400086720376, -0.3010299956639812,
                           -0.2218487496163564, -0.15490195998574313,
                           -0.09691001300805639, -0.045757490560675115]
        num_iterations = 400
        log_p_thermo_beta_sweep = np.full((len(num_particles_list),
                                           len(log_beta_mins_2),
                                           num_iterations), np.nan)

        log_p_non_thermo = np.full((len(non_thermo_train_modes), len(num_particles_list), num_iterations), np.nan)

        train_mode = 'thermo'
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            num_partitions = 2
            for log_beta_min_idx, log_beta_min in enumerate(log_beta_mins_2):
                dir_ = util.get_most_recent_dir_args_match(
                    train_mode=train_mode,
                    architecture=architecture,
                    learning_rate=learning_rate,
                    num_particles=num_particles,
                    num_partitions=num_partitions,
                    log_beta_min=log_beta_min,
                    seed=seed
                )
                if dir_ is not None:
                    stats = util.load_object(util.get_stats_path(dir_))
                    log_p_thermo_beta_sweep[
                        num_particles_idx,
                        log_beta_min_idx] = stats.log_p_history
                    print('thermo {} ({} partitions) beta_min = 1e{} after {}'
                          ' it: {}'.format(
                            num_particles, num_partitions, log_beta_min,
                            len(stats.log_p_history), stats.log_p_history[-1]))
                else:
                    print('missing')

        seed = 7
        log_beta_min = -10
        learning_rate = 3e-4
        num_partitions = 1
        for train_mode_idx, train_mode in enumerate(non_thermo_train_modes):
            for num_particles_idx, num_particles in enumerate(num_particles_list):
                dir_ = util.get_most_recent_dir_args_match(
                    train_mode=train_mode,
                    architecture=architecture,
                    learning_rate=learning_rate,
                    num_particles=num_particles,
                    num_partitions=num_partitions,
                    log_beta_min=log_beta_min,
                    seed=seed
                )
                if dir_ is not None:
                    stats = util.load_object(util.get_stats_path(dir_))
                    log_p_non_thermo[train_mode_idx, num_particles_idx, :len(stats.log_p_history)] = stats.log_p_history

                    print('{} {} after {} it: {}'.format(
                        train_mode, num_particles, len(stats.log_p_history),
                        stats.log_p_history[-1]))
                else:
                    print('missing')

        fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6, 4))

        colors = ['C1', 'C2', 'C4', 'C5']
        linestyles = ['dashed', 'dotted']
        for train_mode_idx, train_mode in enumerate(non_thermo_train_modes):
            for num_particles_idx, num_particles in enumerate(num_particles_list):
                if train_mode == 'ww':
                    label = 'rws'
                else:
                    label = train_mode
                ax.plot(log_p_non_thermo[train_mode_idx, num_particles_idx],
                        linestyle=linestyles[train_mode_idx],
                        color=colors[num_particles_idx],
                        label='{} {} ({:.2f})'.format(
                            label, num_particles,
                            log_p_non_thermo[train_mode_idx, num_particles_idx, -1]))

        # best_num_particles_idx = 3
        # best_beta_idxs = [4, 5, 11]
        # best_beta_idxs = [0, 4, 7, 11]
        best_beta_idxs = [18, 5, 11, 12]
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            best_beta_idx = best_beta_idxs[num_particles_idx]
            color = colors[num_particles_idx]
            ax.plot(log_p_thermo_beta_sweep[num_particles_idx, best_beta_idx],
                    linestyle='solid',
                    color=color,
                    label='thermo S={}, K={}, $\\beta_1$={:.0e} ({:.2f})'.format(
                        num_particles_list[num_particles_idx],
                        2,
                        10**(log_beta_mins_2[best_beta_idx]),
                        log_p_thermo_beta_sweep[num_particles_idx, best_beta_idx, -1]
                    ))

        ax.set_ylim(-110)
        ax.grid(True, axis='y', linewidth=0.2)
        ax.legend(fontsize=6, ncol=3, frameon=False)
        ax.set_ylabel(r'$\log p(x)$')
        ax.set_xlabel('iteration')
        ax.xaxis.set_label_coords(0.5, -0.025)
        ax.set_xticks([0, num_iterations])
        ax.set_xticklabels([0, '4e6'])
        sns.despine(ax=ax, trim=True)

        fig.tight_layout()
        if not os.path.exists('./plots/'):
            os.makedirs('./plots/')
        filename = './plots/baselines.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('saved to {}'.format(filename))
    elif args.mode == 'grad_std':
        learning_rate = 3e-4
        architecture = 'linear_3'
        seed = 8
        non_thermo_train_modes = ['ww', 'vimco']
        num_particles_list = [2, 5, 10, 50]
        num_partitions_list = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        log_beta_mins_1 = [-10, -1, -0.045757490560675115]
        log_beta_mins_2 = [-5, -2, -1.6989700043360187, -1.5228787452803376,
                           -1.3979400086720375, -1.3010299956639813,
                           -1.2218487496163564, -1.1549019599857433,
                           -1.0969100130080565, -1.0457574905606752,
                           -1, -0.6989700043360187, -0.5228787452803375,
                           -0.3979400086720376, -0.3010299956639812,
                           -0.2218487496163564, -0.15490195998574313,
                           -0.09691001300805639, -0.045757490560675115]
        num_iterations = 400
        log_p_thermo_beta_sweep = np.full((len(num_particles_list),
                                           len(log_beta_mins_2),
                                           num_iterations), np.nan)

        log_p_non_thermo = np.full((len(non_thermo_train_modes), len(num_particles_list), num_iterations), np.nan)

        train_mode = 'thermo'
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            num_partitions = 2
            for log_beta_min_idx, log_beta_min in enumerate(log_beta_mins_2):
                dir_ = util.get_most_recent_dir_args_match(
                    train_mode=train_mode,
                    architecture=architecture,
                    learning_rate=learning_rate,
                    num_particles=num_particles,
                    num_partitions=num_partitions,
                    log_beta_min=log_beta_min,
                    seed=seed
                )
                if dir_ is not None:
                    stats = util.load_object(util.get_stats_path(dir_))
                    log_p_thermo_beta_sweep[
                        num_particles_idx,
                        log_beta_min_idx] = stats.grad_std_history
                    print('thermo {} ({} partitions) beta_min = 1e{} after {}'
                          ' it: {}'.format(
                            num_particles, num_partitions, log_beta_min,
                            len(stats.log_p_history), stats.log_p_history[-1]))
                else:
                    print('missing')

        seed = 7
        log_beta_min = -10
        learning_rate = 3e-4
        num_partitions = 1
        for train_mode_idx, train_mode in enumerate(non_thermo_train_modes):
            for num_particles_idx, num_particles in enumerate(num_particles_list):
                dir_ = util.get_most_recent_dir_args_match(
                    train_mode=train_mode,
                    architecture=architecture,
                    learning_rate=learning_rate,
                    num_particles=num_particles,
                    num_partitions=num_partitions,
                    log_beta_min=log_beta_min,
                    seed=seed
                )
                if dir_ is not None:
                    stats = util.load_object(util.get_stats_path(dir_))
                    log_p_non_thermo[train_mode_idx, num_particles_idx, :len(stats.log_p_history)] = stats.grad_std_history

                    print('{} {} after {} it: {}'.format(
                        train_mode, num_particles, len(stats.log_p_history),
                        stats.log_p_history[-1]))
                else:
                    print('missing')

        fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6, 4))

        colors = ['C1', 'C2', 'C4', 'C5']
        linestyles = ['dashed', 'dotted']
        for train_mode_idx, train_mode in enumerate(non_thermo_train_modes):
            for num_particles_idx, num_particles in enumerate(num_particles_list):
                if train_mode == 'ww':
                    label = 'rws'
                else:
                    label = train_mode
                ax.plot(log_p_non_thermo[train_mode_idx, num_particles_idx],
                        linestyle=linestyles[train_mode_idx],
                        color=colors[num_particles_idx],
                        label='{} {} ({:.2f})'.format(
                            label, num_particles,
                            log_p_non_thermo[train_mode_idx, num_particles_idx, -1]))

        # best_num_particles_idx = 3
        # best_beta_idxs = [4, 5, 11]
        # best_beta_idxs = [0, 4, 5, 11]
        best_beta_idxs = [18, 5, 11, 12]
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            best_beta_idx = best_beta_idxs[num_particles_idx]
            color = colors[num_particles_idx]
            ax.plot(log_p_thermo_beta_sweep[num_particles_idx, best_beta_idx],
                    linestyle='solid',
                    color=color,
                    label='thermo S={}, K={}, $\\beta_1$={:.0e} ({:.2f})'.format(
                        num_particles_list[num_particles_idx],
                        2,
                        10**(log_beta_mins_2[best_beta_idx]),
                        log_p_thermo_beta_sweep[num_particles_idx, best_beta_idx, -1]
                    ))

        ax.set_ylim(0, 20)
        ax.grid(True, axis='y', linewidth=0.2)
        ax.legend(fontsize=6, ncol=3, frameon=False)
        ax.set_ylabel(r'grad std')
        ax.set_xlabel('iteration')
        ax.xaxis.set_label_coords(0.5, -0.025)
        ax.set_xticks([0, num_iterations])
        ax.set_xticklabels([0, '4e6'])
        sns.despine(ax=ax, trim=True)

        fig.tight_layout()
        if not os.path.exists('./plots/'):
            os.makedirs('./plots/')
        filename = './plots/grad_std.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('saved to {}'.format(filename))
    elif args.mode == 'baselines_kl':
        learning_rate = 3e-4
        architecture = 'linear_3'
        seed = 8
        non_thermo_train_modes = ['ww', 'vimco']
        num_particles_list = [2, 5, 10, 50]
        num_partitions_list = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        # log_beta_mins_1 = [-10, -1, -0.045757490560675115]
        log_beta_mins_2 = [-5, -2, -1.6989700043360187, -1.5228787452803376,
                           -1.3979400086720375, -1.3010299956639813,
                           -1.2218487496163564, -1.1549019599857433,
                           -1.0969100130080565, -1.0457574905606752,
                           -1, -0.6989700043360187, -0.5228787452803375,
                           -0.3979400086720376, -0.3010299956639812,
                           -0.2218487496163564, -0.15490195998574313,
                           -0.09691001300805639, -0.045757490560675115]
        num_iterations = 400
        log_p_thermo_beta_sweep = np.full((len(num_particles_list),
                                           len(log_beta_mins_2),
                                           num_iterations), np.nan)

        log_p_non_thermo = np.full((len(non_thermo_train_modes), len(num_particles_list), num_iterations), np.nan)

        train_mode = 'thermo'
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            num_partitions = 2
            for log_beta_min_idx, log_beta_min in enumerate(log_beta_mins_2):
                dir_ = util.get_most_recent_dir_args_match(
                    train_mode=train_mode,
                    architecture=architecture,
                    learning_rate=learning_rate,
                    num_particles=num_particles,
                    num_partitions=num_partitions,
                    log_beta_min=log_beta_min,
                    seed=seed
                )
                if dir_ is not None:
                    stats = util.load_object(util.get_stats_path(dir_))
                    log_p_thermo_beta_sweep[
                        num_particles_idx,
                        log_beta_min_idx] = stats.kl_history
                    print('thermo {} ({} partitions) beta_min = 1e{} after {}'
                          ' it: {}'.format(
                            num_particles, num_partitions, log_beta_min,
                            len(stats.log_p_history), stats.log_p_history[-1]))
                else:
                    print('missing')

        seed = 7
        log_beta_min = -10
        learning_rate = 3e-4
        num_partitions = 1
        for train_mode_idx, train_mode in enumerate(non_thermo_train_modes):
            for num_particles_idx, num_particles in enumerate(num_particles_list):
                dir_ = util.get_most_recent_dir_args_match(
                    train_mode=train_mode,
                    architecture=architecture,
                    learning_rate=learning_rate,
                    num_particles=num_particles,
                    num_partitions=num_partitions,
                    log_beta_min=log_beta_min,
                    seed=seed
                )
                if dir_ is not None:
                    stats = util.load_object(util.get_stats_path(dir_))
                    log_p_non_thermo[train_mode_idx, num_particles_idx, :len(stats.kl_history)] = stats.kl_history

                    print('{} {} after {} it: {}'.format(
                        train_mode, num_particles, len(stats.log_p_history),
                        stats.log_p_history[-1]))
                else:
                    print('missing')

        fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6, 4))

        colors = ['C1', 'C2', 'C4', 'C5']
        linestyles = ['dashed', 'dotted']
        for train_mode_idx, train_mode in enumerate(non_thermo_train_modes):
            for num_particles_idx, num_particles in enumerate(num_particles_list):
                if train_mode == 'ww':
                    label = 'rws'
                else:
                    label = train_mode
                ax.plot(log_p_non_thermo[train_mode_idx, num_particles_idx],
                        linestyle=linestyles[train_mode_idx],
                        color=colors[num_particles_idx],
                        label='{} {} ({:.2f})'.format(
                            label, num_particles,
                            log_p_non_thermo[train_mode_idx, num_particles_idx, -1]))

        # best_num_particles_idx = 3
        # best_beta_idxs = [4, 5, 11]
        # best_beta_idxs = [0, 4, 5, 11]
        best_beta_idxs = [18, 5, 11, 12]
        for num_particles_idx, num_particles in enumerate(num_particles_list):
            best_beta_idx = best_beta_idxs[num_particles_idx]
            color = colors[num_particles_idx]
            ax.plot(log_p_thermo_beta_sweep[num_particles_idx, best_beta_idx],
                    linestyle='solid',
                    color=color,
                    label='thermo S={}, K={}, $\\beta_1$={:.0e} ({:.2f})'.format(
                        num_particles_list[num_particles_idx],
                        2,
                        10**(log_beta_mins_2[best_beta_idx]),
                        log_p_thermo_beta_sweep[num_particles_idx, best_beta_idx, -1]
                    ))

        ax.set_ylim(5, 20)
        ax.grid(True, axis='y', linewidth=0.2)
        ax.legend(fontsize=6, ncol=3, frameon=False)
        ax.set_ylabel(r'KL(q || p)')
        ax.set_xlabel('iteration')
        ax.xaxis.set_label_coords(0.5, -0.025)
        ax.set_xticks([0, num_iterations])
        ax.set_xticklabels([0, '4e6'])
        sns.despine(ax=ax, trim=True)

        fig.tight_layout()
        if not os.path.exists('./plots/'):
            os.makedirs('./plots/')
        filename = './plots/baselines_kl.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('saved to {}'.format(filename))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', default='efficiency',
                        help='efficiency, insights, baselines, baselines_kl or'
                        ' grad_std')
    args = parser.parse_args()
    main(args)
