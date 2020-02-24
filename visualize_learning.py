import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

nn_path = sys.argv[1]

nn_dict = pickle.load(open(nn_path, 'rb'))

gens = []
best = []
means = []
medians = []
stds = []


for data_dict in nn_dict['log']:
    gens.append(data_dict['gen'])
    best.append(data_dict['fit_best'])
    means.append(data_dict['fit_mean'])
    medians.append(data_dict['fit_med'])
    stds.append(data_dict['fit_std'])


print(gens)
print(best)

# Graph data
fig = plt.figure(figsize=(13,7))
ax = plt.axes()
ax.set_ylabel("Fitness", fontsize=15)
ax.set_xlabel("Generations", fontsize=15)
ax.plot(gens, best)
ax.errorbar(gens, medians, stds, label='Best Raw Fitness', fmt='-b', capsize=3)
ax.legend()
fig.suptitle("Fitness Graph", fontsize=25)

plt.show()