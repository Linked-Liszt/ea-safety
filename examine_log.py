import pickle

while True:
    path = 'checkpoints/hparam_search_2/' + input("input_path: ")

    nn_dict = pickle.load(open(path, 'rb'))

    #Breakpoint here
    print()