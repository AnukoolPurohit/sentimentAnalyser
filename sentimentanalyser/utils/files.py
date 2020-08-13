import pickle

def pickle_dump(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj,file)

def pickle_load(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj