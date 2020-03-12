import pickle
import numpy as np
import copy



class EvoACLogger(object):
    def __init__(self, config):

        self.config_exp = config['experiment']
        self.config_evoac = config['evo_ac']

        self.directory = self.config_exp['log_path']
        self.name = self.config_exp['log_name']

        self.print_interval = self.config_exp['print_interval']
        
        self.env = self.config_exp['env']
        
        self.experiment_log = []
        self.run_log = []
        self.run_counter = 0
        self.best_fitness = float('-inf')


    def save_fitnesses(self, model, fitnesses, gen):
        data_dict = {}
        data_dict['gen'] = gen
        data_dict['fit'] = copy.deepcopy(fitnesses)
        data_dict['fit_best'] = np.max(fitnesses)
        data_dict['fit_mean'] = np.mean(fitnesses)
        data_dict['fit_med'] = np.median(fitnesses)
        data_dict['fit_std'] = np.std(fitnesses)
        self.run_log.append(data_dict)

        if float(np.max(fitnesses)) > self.best_fitness:
            self.best_model =  copy.deepcopy(model)

    def end_run(self):
        self.experiment_log.append(self.run_log)
        self.run_log = []
        self._export_data(f'run_{self.run_counter:02d}')

        
    def _export_data(self, export_name):
        data_path = self.directory + '/' + self.name + '_' \
                + export_name + '.p'

        save_dict = {}
        save_dict['env'] = self.env
        save_dict['best_nn'] = self.best_model
        save_dict['experiment_log'] = self.experiment_log
        pickle.dump(save_dict, open(data_path, 'wb'))

    def print_data(self, gen_idx):
        if gen_idx % self.print_interval == 0:
            


