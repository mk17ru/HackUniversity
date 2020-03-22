import pickle
import numpy as np

filename = 'network_model.sav'
arr = [[str(string).__hash__() for string in np.genfromtxt("test.txt", dtype=str)]]

load_lr_model = pickle.load(open(filename, 'rb'))
print(list(map(int, *load_lr_model.predict(arr).tolist())))
