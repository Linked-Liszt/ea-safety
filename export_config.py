import pickle
import sys
import json


in_path = sys.argv[1]
out_path = sys.argv[2]

nn_dict = pickle.load(open(in_path, 'rb'))

config = nn_dict['config']

with open(out_path, 'w') as json_file:
    json.dump(config, json_file, indent=4, sort_keys=True)