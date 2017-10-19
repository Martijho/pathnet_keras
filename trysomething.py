from pathnet_keras import PathNet
from path_search import PathSearch
from analytic import Analytic
from dataprep import DataPrep
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers.merge import add
from keras.optimizers import Adagrad, Adam, RMSprop
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os
import random
import time
import resource
#print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

data = DataPrep()
data.cifar10()

width = 5
depth = 3

x, y, x_test, y_test = data.x, data.y, data.x_test, data.y_test
pn = PathNet(input_shape=x[0].shape, output_size=10, width=width, depth=depth, max_active_modules=75)
ps = PathSearch(pn)

best, history = ps.evolutionary_search(x, y, population_size=6, generations=5)
pn.save_pathnet()
Analytic(pn).process_evolution_run(history)
