from pathnet_keras import PathNet
from path_search import PathSearch
from analytic import Analytic
from dataprep import DataPrep
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import numpy as np
import time


data = DataPrep()
data.mnist()

width = 2
depth = 2

x, y, x_test, y_test = data.sample_dataset([1, 2])#, 3, 5, 7])
#x, y, x_test, y_test = data.sample_dataset([0, 4, 6, 8, 9])


pn = PathNet(input_shape=data.input_shape, output_size=y.shape[1], width=width, depth=depth)
ps = PathSearch(pn)
al = Analytic(pn)




## TODO:
# PRØV Å TRENE, RESETE, TRENE
p = pn.random_path()

pn.train_path(x, y, epochs=1, batch_size=128, path=p)
print(pn.evaluate_path(x_test, y_test, p))

print(pn.weights[0][0][0].get_weights())
import keras
keras.backend.clear_session()
print(pn.weights[0][0][0].get_weights())

print(pn.evaluate_path(x_test, y_test, p))
pn.train_path(x, y, epochs=10, batch_size=128, path=p)
print(pn.evaluate_path(x_test, y_test, p))
'''
path, history = ps.evolutionary_search(x, y, population_size=4, generations=2, clear_session_every=1)
pn = ps.pathnet
al.process_evolution_run(history)

print(pn.evaluate_path(x_test, y_test, path))
pn.print_training_counter()
'''
