import pickle

while True:
    path = 'checkpoints/hparam_search/' + input("input_path: ")

    nn_dict = pickle.load(open(path, 'rb'))

    #Breakpoint here
    print()