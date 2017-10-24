from pathnet_keras import PathNet
from path_search import PathSearch
from analytic import Analytic
from dataprep import DataPrep
from pathnet_constructor import PathNetConstructor
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
data.mnist()

x, y, x_test, y_test = data.sample_dataset([5, 6])
pn = PathNet(input_shape=[28, 28, 1], depth=3, width=10)
pn._binary_mnist_test()
an = Analytic(pn)
ps = PathSearch(pn)

path, history = ps.binary_mnist_tournamet_search(x, y)

