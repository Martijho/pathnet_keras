from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers.merge import add
from keras.optimizers import Adagrad, Adam, RMSprop
from keras.callbacks import TensorBoard
from keras import backend as K
from datetime import datetime
import numpy as np
import pickle
import os
import random
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PathNetConstructor():
    def __init__(self, in_size):
        self.structure = ['input']
        self.thread = Input(in_size)
        self.layers = [self.thread]
        self.names = []
        self.saved_layers = []

    def get_layers(self):
        if len(self.layers) >= 2:
            model = Model(inputs=self.layers[0], outputs=self.layers[-1])
            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

            saved_layers = {}
            for name in self.names:
                assert model.get_layer(name) is not None, 'Created pathnet without correct layers'
                saved_layers[name] = model.get_layer(name)
            self.saved_layers = saved_layers
            return saved_layers, model

    def get_layers_and_unique(self, output_size, unique_name):

        unique = Dense(output_size, activation='softmax', name=unique_name)(self.layers[-1])

        model = Model(inputs=self.layers[0], outputs=unique)
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        saved_layers = {}
        for name in self.names:
            assert model.get_layer(name) is not None, 'Created pathnet without correct layers'
            saved_layers[name] = model.get_layer(name)
        self.saved_layers = saved_layers

        return saved_layers, model.get_layer(unique_name)

    def add_dense_layer(self, width, config):
        if self.structure[-1] == 'conv2d':
            self.thread = MaxPooling2D()(self.thread)
        if self.structure[-1] != 'dense':
            self.thread = Flatten(name='flatten')(self.thread)

        outputs = []
        for m in range(width):
            outputs.append(self._build_dense_module(len(self.layers)-1, m, self.thread, config))

        if width > 1:
            self.thread = add(outputs)
        else:
            self.thread = outputs[0]
        self.layers.append(self.thread)
        self.structure.append('dense')

    def add_conv_layer(self, width, config):

        outputs = []
        for m in range(width):
            outputs.append(self._build_conv_module(len(self.layers)-1, m, self.thread, config))

        if width > 1:
            self.thread = add(outputs)
        else:
            self.thread = outputs[0]
        self.layers.append(self.thread)

        self.structure.append('conv2d')

    def add_maxpool(self):
        pass

    def _build_dense_module(self, layer_index, module_index, module_input, config):
        thread = module_input
        for n, node in enumerate(config):
            thread = Dense(node['out'], activation=node['activation'],
                            name=self._name(layer_index, module_index, n, 'D'))(thread)
        return thread

    def _build_conv_module(self, layer_index, module_index, module_input, config):
        thread = module_input
        for n, node in enumerate(config):
            thread = Conv2D(node['channels'], node['kernel'], activation=node['activation'],
                            name=self._name(layer_index, module_index, n, 'C'))(thread)
        thread = BatchNormalization(name=self._name(layer_index, module_index, 0, 'BN'))(thread)
        return thread

    def _name(self, l, m, n, node_type):
        name = 'L'+str(l)+'M'+str(m)+node_type+str(n)
        if node_type == 'BN':
            name = name[:-1*len(str(n))]
        self.names.append(name)
        return name