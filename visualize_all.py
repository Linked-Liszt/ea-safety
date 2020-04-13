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
    len_runs = 500

    gens = []
    best = []
    means = []
    medians = []
    pop_stds = []
    timesteps = []


    for run_idx, run_log in enumerate(nn_dict['experiment_log']):
        # I'm sorry, this is very ugly
        gens_run = []
        best_run = []
        means_run = []
        medians_run = []
        pop_stds_run = []
        timesteps_run = []
        for gen_idx, data_dict in enumerate(run_log):
            gens_run.append(gen_idx)
            if 'test_fit' in data_dict:
                fit_val = 'test_fit'
            else:
                fit_val = 'fit_best'
                print("WARNING: Test fit not found!")
            best_run.append(data_dict[fit_val])
            means_run.append(data_dict['fit_mean'])
            medians_run.append(data_dict['fit_med'])
            pop_stds_run.append(data_dict['fit_std'])
            timesteps_run.append(data_dict['timesteps'])
        
        interp_points = np.arange(0, nn_dict['config']['experiment']['timesteps'] + 1, step=1000)
    
        best_interp = np.interp(interp_points, timesteps_run, best_run, left=0.0, right=best_run[-1])

        best.append(best_interp)

    std = np.std(best, axis=0)
    best = np.mean(best, axis=0)
    
    print("Log Loaded.")
    return interp_points, best, std


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
    timesteps = []
    bests = []
    stds = []
    for path in log_paths:
        nn_dict = pickle.load(open(path, 'rb'))
    
        timestep, best, std = load_data_v0(nn_dict)

        if parser.ignore_failed and best[-1] < 50:
            continue

        plotted_log_paths.append(path)
        timesteps.append(timestep)
        bests.append(best)
        stds.append(std)    




    # Graph data
    fig, ax_h = plt.subplots(1, 1, figsize=(13,7))
    ax_h.set_ylabel("Fitness", fontsize=15)
    ax_h.set_xlabel("Timesteps", fontsize=15)


    for path_idx, path in enumerate(plotted_log_paths):
        print(path)
        ax_h.plot(timesteps[path_idx], bests[path_idx], label=get_log_name(path))
        error_high = stds[path_idx] + bests[path_idx]
        error_low = (-stds[path_idx]) + bests[path_idx]
        ax_h.fill_between(timesteps[path_idx], error_low, error_high, alpha=0.2)
        


    ax_h.legend(loc='lower right', shadow=True, fontsize=14)
    fig.suptitle("Cartpole Ablation", fontsize=25)

    plt.show()