"""
 A library to load the MNIST image data.
"""

import pickle
import gzip
import numpy as np

def get_path(path2data):
    path2code = __file__.split("/")
    path2data = path2data.split("/")
    
    path2code.pop(-1)
    
    while path2data[0] == "..":
        path2data.pop(0)
        path2code.pop(-1)
        
    if len(path2code) != 0:
        path = "/".join(path2code) + "/" + "/".join(path2data)
    else:
        path = "/".join(path2data)
    
    return path

def load_data():
    path2data = "../data/mnist.pkl.gz"
    path = get_path(path2data)
    f = gzip.open(path)
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return training_data, validation_data, test_data

def load_data_wrapper():
    tr_d,va_d,te_d = load_data()
    
    training_data = [np.reshape(x,(784,1)) for x in tr_d[0]]
    training_label = [vectorize_label(i) for i in tr_d[1]]
    training_data = (training_data,training_label)
    
    validation_data = [np.reshape(x,(784,1)) for x in va_d[0]]
    validation_data = (validation_data,va_d[1])
    
    test_data = [np.reshape(x,(784,1)) for x in te_d[0]]
    test_data = (test_data,te_d[1])
    
    return training_data, validation_data, test_data
    

"""
    Return a 10-dimensional unit vector with a 1.0 in the jth
position and zeroes elsewhere.  This is used to convert a digit
(0...9) into a corresponding desired output from the neural
network.
"""
def vectorize_label(i):
    vec_label = np.zeros((10,1))
    vec_label[i] = 1.0
    return vec_label
    

    
    
    
    
    
    






