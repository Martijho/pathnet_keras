from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers.merge import add
from keras.optimizers import Adagrad, Adam, RMSprop
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras import backend as K
import numpy as np
import pickle
import os
import random
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class PathNet:
    def __init__(self, input_shape=None, output_size=1, width=-1, depth=-1, load_file=False, max_active_modules=75):
        self._models_created_in_current_session = 0
        self.tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,write_images=True)
        self._max_active_modules = max_active_modules
        if load_file:
            self.load_pathnet()
            return

        self.optimal_paths = []
        self.training_counter = [[0]*width for _ in range(depth)]
        self.depth = depth
        self.width = width
        self.input_shape = input_shape
        self.output_size = output_size

        self.dense_module_config = [{'out': output_size, 'activation': 'softmax'}]            # Add another dict to inc
        self.conv_module_config = [{'channels': 8, 'kernel': (3,3), 'activation': 'relu'},
                                   {'channels': 3, 'kernel': (3,3), 'activation': 'relu'},
                                   {'channels': 1, 'kernel': (3,3), 'activation': 'relu'}]     # Add another dict to inc
        self.conv_module_includes_batchnorm = True

        self._init_whole_pathnet()

    def _name(self, l, m, n, node_type):
        name = 'L'+str(l)+'M'+str(m)+node_type+str(n)
        if node_type == 'BN':
            name = name[:-1*len(str(n))]
        return name

    def _init_whole_pathnet(self):

        inp = Input(self.input_shape)
        thread = inp

        for l in range(self.depth):
            layer_outputs = []
            for m in range(self.width):
                node_thread = thread
                if l == self.depth-1:
                    layer_outputs.append(self._build_dense_module(l, m, node_thread))
                else:
                    layer_outputs.append(self._build_conv_module(l, m, node_thread))

            if len(layer_outputs) == 1:
                thread = layer_outputs[0]
            else:
                thread = add(layer_outputs)

            if l == self.depth-2:
                thread = MaxPooling2D((2, 2))(thread)
                thread = Flatten()(thread)


        model = Model(inputs=inp, outputs=thread)
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        self._models_created_in_current_session +=1
        self._saved_layers = {}
        for n in self.path2layer_names([list(range(self.width))]*self.depth):
            self._saved_layers[n] = model.get_layer(n)

    def load_weights_to_model(self, model):
        for name, layer in self.weights.items():
            if model.get_layer(name) is not None:
                model.get_layer(name).set_weights(layer.get_weights())

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

    def path2model(self, path, optimizer='adam', loss='categorical_crossentropy'):
        self._models_created_in_current_session += 1
        if self._models_created_in_current_session >= self._max_active_modules:
            self.reset_backend_session()

        inp = Input(self.input_shape)
        thread = inp

        for l, active_modules in enumerate(path):
            layer_outputs = []
            for m in active_modules:
                module_thread = thread
                if l == self.depth-1:
                    for d in range(len(self.dense_module_config)):
                        name = self._name(l, m, d, 'D')
                        module_thread = self._saved_layers[name](module_thread)
                else:
                    for c in range(len(self.conv_module_config)):
                        name = self._name(l, m, c, 'C')
                        module_thread = self._saved_layers[name](module_thread)
                    if self.conv_module_includes_batchnorm:
                        name = self._name(l, m, 0, 'BN')
                        module_thread = self._saved_layers[name](module_thread)
                layer_outputs.append(module_thread)

            if len(layer_outputs) == 1:
                thread = layer_outputs[0]
            else:
                thread = add(layer_outputs)

            if l == self.depth-2:
                thread = MaxPooling2D((2, 2))(thread)
                thread = Flatten()(thread)

        m = Model(inputs=inp, outputs=thread)
        m.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return m

    def _build_dense_module(self, layer_index, module_index, module_input):
        thread = module_input
        for n, node in enumerate(self.dense_module_config):
            thread = Dense(node['out'], activation=node['activation'],
                            name=self._name(layer_index, module_index, n, 'D'))(thread)
        return thread

    def _build_conv_module(self, layer_index, module_index, module_input):
        thread = module_input
        for n, node in enumerate(self.conv_module_config):
            thread = Conv2D(node['channels'], node['kernel'], activation=node['activation'],
                            name=self._name(layer_index, module_index, n, 'C'))(thread)
        if self.conv_module_includes_batchnorm:
            thread = BatchNormalization(name=self._name(layer_index, module_index, 0, 'BN'))(thread)
        return thread

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
            'out_size': self.output_size,
            'training_counter': self.training_counter,
            'dense_module_config': self.dense_module_config,
            'conv_module_config': self.conv_module_config,
            'conv_module_includes_batchnorm':self.conv_module_includes_batchnorm,
            'trainable': trainable,
            'weights': weights
            }

        with open('logs/log_dict.pkl', 'wb') as f:
            pickle.dump(log, f)

    def load_pathnet(self):
        log = None
        with open('logs/log_dict.pkl', 'rb') as f:
            log = pickle.load(f)

        self.optimal_paths = log['optimal_paths']
        self.depth = log['depth']
        self.width = log['width']
        self.input_shape = log['in_shape']
        self.output_size = log['out_size']
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
        t = time.time()
        for name, layer in self._saved_layers.items():
                weights[name] = layer.get_weights()
                trainable[name] = layer.trainable

        K.clear_session()
        self._models_created_in_current_session = 0
        self._init_whole_pathnet()

        for name, layer in self._saved_layers.items():
            layer.set_weights(weights[name])
            layer.trainable = trainable[name]
