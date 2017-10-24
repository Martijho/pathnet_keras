from pathnet_constructor import PathNetConstructor
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers.merge import add
from keras.optimizers import Adagrad, Adam, RMSprop, SGD
from keras.callbacks import TensorBoard
from keras import backend as K
from datetime import datetime
import numpy as np
import pickle
import os
import random
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class PathNet:
    def __init__(self, input_shape=None, width=-1, depth=-1, load_file=False, max_active_modules=75):
        self._models_created_in_current_session = 0
        self.tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,write_images=True)
        self._max_active_modules = max_active_modules

        if load_file:
            self.load_pathnet()
            return

        self.optimal_paths = []
        self.output_sizes = []
        self.unique_classification_layer = []

        self.training_counter = [[0]*width for _ in range(depth)]
        self.depth = depth
        self.width = width
        self.input_shape = input_shape
        self._saved_layers = None

        #self._binary_mnist_test()
        ''''
        self.dense_module_config = [{'out': 20, 'activation': 'relu'}]            # Add another dict to inc
        self.conv_module_config = [{'channels': 3, 'kernel': (5,5), 'activation': 'relu'},
                                   {'channels': 1, 'kernel': (3,3), 'activation': 'relu'}]     # Add another dict to inc
        self.conv_module_includes_batchnorm = True


        #self._init_whole_pathnet()
        '''

    def _name(self, l, m, n, node_type):
        name = 'L'+str(l)+'M'+str(m)+node_type+str(n)
        if node_type == 'BN':
            name = name[:-1*len(str(n))]
        return name

    def _binary_mnist_test(self):
        self.dense_module_config = [{'out': 20, 'activation': 'relu'}]
        self.depth = 3
        self.width = 10
        self.max_modules_pr_layer = 3

        self.pathnet_dimentions = [1, 1, 1]
        self.maxpool_placement = -1
        self.flatten_placement = 0

        self.learning_rate = 0.0001
        self.optimizer_type = SGD
        self.loss = 'categorical_crossentropy'

        if self._saved_layers is None:
            pnc = PathNetConstructor([28, 28, 1])
            pnc.add_dense_layer(self.width, self.dense_module_config)
            pnc.add_dense_layer(self.width, self.dense_module_config)
            pnc.add_dense_layer(self.width, self.dense_module_config)

            self._saved_layers, unique = pnc.get_layers_and_unique(2, 'binary_mnist_unique')
            self._models_created_in_current_session += 1
        else:
            if 'L'+str(self.depth-1)+'M'+str(0) + 'D' + str(self.pathnet_dimentions[-1]-1) in self._saved_layers:
                last_layer = self._saved_layers['L'+str(self.depth-1)+'M'+str(0)+'D'+str(self.pathnet_dimentions[-1]-1)]
            elif 'L'+str(self.depth-1)+'M'+str(0)+'C'+str(self.pathnet_dimentions[-1]-1) in self._saved_layers:
                last_layer = self._saved_layers['L'+str(self.depth-1)+'M'+str(0)+'C'+str(self.pathnet_dimentions[-1] - 1)]
            else:
                last_layer = self._saved_layers['L'+str(self.depth-1)+'M'+str(0) + 'BN']

            tmp_inp = Input(last_layer.compute_output_shape([28, 28, 1]))
            unique = Dense(2, activation='softmax', name='binary_mnist_unique')(tmp_inp)
            tmp_model = Model(inputs=tmp_inp, outputs=unique)
            tmp_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
            unique = tmp_model.get_layer('binary_mnist_unique')
            self._models_created_in_current_session += 2

        self.output_sizes.append(2)
        self.unique_classification_layer.append(unique)

    def _cifar_test(self):
        self.dense_module_config = [{'out': 20, 'activation': 'relu'}]
        self.depth = 3
        self.width = 20
        self.max_modules_pr_layer = 5

        pnc = PathNetConstructor([32, 32, 3])
        pnc.add_dense_layer(self.width, self.dense_module_config)
        pnc.add_dense_layer(self.width, self.dense_module_config)
        pnc.add_dense_layer(self.width, self.dense_module_config)

        self.pathnet_dimentions = [1, 1, 1]

        self._saved_layers = pnc.get_layers()

    def path2layer_names(self, path):
        names = []
        for l, layer in enumerate(path):
            for m in layer:
                prefix = 'L'+str(l)+'M'+str(m)
                if l == len(path)-1:
                    for d in range(len(self.dense_module_config)):
                        names.append(prefix+'D'+str(d))
                else:
                    for c in range(len(self.conv_module_config)):
                        names.append(prefix+'C' + str(c))
                    if self.conv_module_includes_batchnorm:
                        names.append(prefix+'BN')
        return names

    def path2model(self, path, task_nr=0):
        self._models_created_in_current_session += 1
        if self._models_created_in_current_session >= self._max_active_modules:
            self.reset_backend_session()

        inp = Input(self.input_shape)
        thread = inp

        for l in range(self.depth):
            layer_prefix = 'L'+str(l)
            layer_outputs = []

            if self.flatten_placement == l:
                thread = Flatten()(thread)

            for m in path[l]:
                module_prefix = 'M'+str(m)
                module_thread = thread

                for n in range(self.pathnet_dimentions[l]):
                    layer = None
                    if layer_prefix+module_prefix+'D'+str(n) in self._saved_layers.keys():
                        layer = self._saved_layers[layer_prefix+module_prefix+'D'+str(n)]
                    elif layer_prefix+module_prefix+'C'+str(n) in self._saved_layers.keys():
                        layer = self._saved_layers[layer_prefix+module_prefix+'C'+str(n)]

                    module_thread = layer(module_thread)

                if layer_prefix+module_prefix+'BN' in self._saved_layers.keys():
                    module_thread = self._saved_layers[layer_prefix+module_prefix+'BN'](module_thread)

                layer_outputs.append(module_thread)

            if len(layer_outputs) > 1:
                thread = add(layer_outputs)
            else:
                thread = layer_outputs[0]

            if self.maxpool_placement == l:
                thread = MaxPooling2D((2, 2))(thread)

        thread = self.unique_classification_layer[task_nr](thread)

        model = Model(inputs=inp, outputs=thread)
        optimizer = self.optimizer_type(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])

        return model

    def _lock_modules_from_training(self, path):
        for name in self.path2layer_names(path):
            self._saved_layers[name].trainable = False

    def _reinitialize_open_layers(self, sigma=0.5):
        ### NOTE:
        ### During curriculum, data is from the same distribution, and reinitialization of batchnormalization layers
        ### is unnecessary

        for name, layer in self._saved_layers.items():
            if layer.trainable:
                if type(layer) is BatchNormalization:
                    layer.set_weights([
                        np.ones_like(layer.get_weights()[0]),
                        np.zeros_like(layer.get_weights()[1]),
                        np.zeros_like(layer.get_weights()[2]),
                        np.ones_like(layer.get_weights()[3])
                    ])
                else:
                    layer.set_weights([
                        np.random.normal(0, sigma, size=layer.get_weights()[0].shape),
                        np.zeros_like(layer.get_weights()[1])
                    ])

    def save_new_optimal_path(self, path):
        self.optimal_paths.append(path)
        self._lock_modules_from_training(path)
        self._reinitialize_open_layers()

        new_counter = np.zeros_like(self.training_counter)

        for p in self.optimal_paths:
            for layer in range(self.depth):
                for module in p[layer]:
                    new_counter[layer][module] = self.training_counter[layer][module]

        self.training_counter = new_counter

    def random_path(self, min=1, max=2):
        if min < 1:
            min = 1

        max = self.max_modules_pr_layer
        path = []
        for _ in range(self.depth):
            contenders = list(range(self.width))
            np.random.shuffle(contenders)
            if min == max:
                path.append(contenders[:min])
            else:
                path.append(contenders[:np.random.randint(low=min, high=max+1)])

        return path
        
    def increment_training_counter(self, path):
        for layer in range(self.depth):
            for module in path[layer]:
                self.training_counter[layer][module]+=1

    def save_pathnet(self):
        trainable = {}
        weights = {}
        for name, layer in self._saved_layers.items():
            trainable[name] = layer.trainable
            weights[name] = layer.get_weights()

        log = {
            'optimal_paths': self.optimal_paths,
            'depth': self.depth,
            'width': self.width,
            'in_shape': self.input_shape,
            'out_sizes': self.output_sizes,
            'unique_classification_weights': self.unique_classification_weights,
            'training_counter': self.training_counter,
            'dense_module_config': self.dense_module_config,
            'conv_module_config': self.conv_module_config,
            'conv_module_includes_batchnorm':self.conv_module_includes_batchnorm,
            'trainable': trainable,
            'weights': weights
            }

        now = datetime.now()
        with open('logs/PNstate '+str(now)[:16]+'.pkl', 'wb') as f:
            pickle.dump(log, f)

    def load_pathnet(self, filename):
        log = None
        with open('logs/'+filename, 'rb') as f:
            log = pickle.load(f)

        self.optimal_paths = log['optimal_paths']
        self.depth = log['depth']
        self.width = log['width']
        self.input_shape = log['in_shape']
        self.output_sizes = log['out_sizes']
        self.unique_classification_weights = log['unique_classification_weights']
        self.training_counter = log['training_counter']
        self.dense_module_config = log['dense_module_config']
        self.conv_module_config = log['conv_module_config']
        self.conv_module_includes_batchnorm = log['conv_module_includes_batchnorm']

        self._init_whole_pathnet()

        for name, layer in self._saved_layers.items():
            layer.set_weights(log['weights'][name])
            layer.trainable = log['trainable'][name]

    def train_path(self, x, y, path=None, epochs=20, batch_size=64, verbose=True, model=None, callback=None, validation_split=0.2):
        if path is None:
            path = self.random_path(max=3)
        if model is None:
            model = self.path2model(path)

        if callback is None:
            hist = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split)
        else:
            hist = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.2, callbacks=[callback])

        for _ in range(epochs):
            self.increment_training_counter(path)

        return model, path, hist.history

    def evaluate_path(self, x, y, path):
        model = self.path2model(path)
        predictions = model.predict(x)

        hit = 0
        miss = 0
        for p, t in zip(predictions, y):
            if np.argmax(p) == np.argmax(t):
                hit+=1
            else:
                miss+=1

        return hit/(hit+miss)

    def reset_backend_session(self):
        print('==> Reseting backend session')
        weights = {}
        trainable = {}

        unique_weights = []
        for w in self.unique_classification_layer:
            unique_weights.append(w.get_weights())

        for name, layer in self._saved_layers.items():
                weights[name] = layer.get_weights()
                trainable[name] = layer.trainable

        K.clear_session()
        self._models_created_in_current_session = 0
        self._init_whole_pathnet()

        self.unique_classification_layer = []
        for ind, weights in enumerate(unique_weights):
            layer = Dense(self.output_sizes[ind], activation='softmax')
            layer.set_weights(weights)
            self.unique_classification_layer.append(layer)

        for name, layer in self._saved_layers.items():
            layer.set_weights(weights[name])
            layer.trainable = trainable[name]


