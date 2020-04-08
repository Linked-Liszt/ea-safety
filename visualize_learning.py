import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import argparse
import os


def load_data_v0(log_dict):
    print("Loading log...")
    num_runs = len(nn_dict['experiment_log'])
    len_runs = len(nn_dict['experiment_log'][0])

    gens = np.arange(0, len_runs)
    best = np.zeros((num_runs, len_runs))
    means = np.zeros((num_runs, len_runs))
    medians = np.zeros((num_runs, len_runs))
    pop_stds = np.zeros((num_runs, len_runs))


    for run_idx, run_log in enumerate(nn_dict['experiment_log']):
        for gen_idx, data_dict in enumerate(run_log):
            best[run_idx][gen_idx] = data_dict['fit_best']
            means[run_idx][gen_idx] = data_dict['fit_mean']
            medians[run_idx][gen_idx] = data_dict['fit_med']
            pop_stds[run_idx][gen_idx] = data_dict['fit_std']

    episodes = np.array(gens) * log_dict['config']['evo_ac']['pop_size']

    
    best = np.mean(best, axis=0)
    means = np.mean(means, axis=0)
    medians = np.mean(medians, axis=0)
    pop_stds = np.mean(pop_stds, axis=0)
    stds = np.std(means, axis=0)
    print("Log Loaded.")
    return episodes, best, means, medians, pop_stds, stds


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_paths', metavar='log files or folders', type=str, nargs='+',
                         help='log files or folder to scan')

    parser.add_argument('--f', action='store_true', dest='folder_flag',
                         help='scan entire folder of logs')

    parser.add_argument('--i', action='store_true', dest='ignore_failed',
                         help='ignore runs that didn\'t solve the task')

    return parser.parse_args()

def scan_folder(folder_paths):
    log_files = []
    for folder_path in folder_paths:
        for file in os.listdir(folder_path):
            if file.endswith('.p'):
                log_files.append(os.path.join(folder_path, file))
    return log_files

def get_log_name(path):
    log_name = path
    if '\\' in path:
        log_name = path[path.rfind('\\') + 1:path.rfind('.')]
    elif '/' in path:
        log_name = path[path.rfind('/') + 1:path.rfind('.')]
    return log_name

if __name__ == '__main__':
    parser = parse_arguments()

    if parser.folder_flag:
        print(parser.log_paths)
        log_paths = scan_folder(parser.log_paths)
    else:
        log_paths = parser.log_paths

    plotted_log_paths = []
    episodes = []
    bests = []
    means = []
    medians = []
    pop_stds = []
    stds = []
    for path in log_paths:
        nn_dict = pickle.load(open(path, 'rb'))
    
        episode, best, mean, median, pop_std, std = load_data_v0(nn_dict)

        if parser.ignore_failed and best[-1] < 50:
            continue

        plotted_log_paths.append(path)
        episodes.append(episode)
        bests.append(best)
        means.append(mean)
        medians.append(median)
        pop_stds.append(pop_std)
        stds.append(std)


    # Graph data
    fig, (ax_h, ax_l) = plt.subplots(2, 1, figsize=(13,7), sharex=True)
    ax_h.set_ylabel("Best Fitness", fontsize=15)
    ax_l.set_ylabel("Population Fitness", fontsize=15)
    ax_l.set_xlabel("Episodes", fontsize=15)

    for path_idx, path in enumerate(plotted_log_paths):
        log_name = get_log_name(path)

        ax_h.plot(episodes[path_idx], bests[path_idx], label=log_name)
        ax_l.plot(episodes[path_idx], means[path_idx], label=log_name)

    ax_l.legend(loc='upper center', bbox_to_anchor=(1.05, 1.5), shadow=True, fontsize=7)
    #ax_l.legend()
    fig.suptitle("Fitness Graph", fontsize=25)

    plt.show()