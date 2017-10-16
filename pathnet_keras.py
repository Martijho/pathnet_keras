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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class PathNet:
    def __init__(self, input_shape=None, output_size=1, width=-1, depth=-1, load_file=False):

        self.tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,write_images=True)
        self.weights = None
        self.max_number_of_modules_in_layer = None
        self.min_number_of_modules_in_layer = None

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
        self.conv_module_config = [{'channels': 3, 'kernel': (3,3), 'activation': 'relu'},
                                   {'channels': 1, 'kernel': (3,3), 'activation': 'relu'}]     # Add another dict to inc
        self.conv_module_includes_batchnorm = True

        self._init_pathnet_weights()
        self.build_model_from_path()

       # self._reinitialize_open_layers_uniformly()

    def _init_pathnet_weights(self):
        weights = []
        for l in range(self.depth):
            layer = []
            for m in range(self.width):
                module = []
                if l == self.depth-1:
                    for node_ind, node_dict in enumerate(self.dense_module_config):
                        node = Dense(node_dict['out'], activation=node_dict['activation'],
                                     name='L' + str(l) + 'M' + str(m) + 'D' + str(node_ind))
                        module.append(node)
                else:
                    for node_ind, node_dict in enumerate(self.conv_module_config):
                        node = Conv2D(node_dict['channels'], node_dict['kernel'], activation=node_dict['activation'],
                                      name='L' + str(l) + 'M' + str(m) + 'C' + str(node_ind))
                        module.append(node)
                    if self.conv_module_includes_batchnorm:
                        module.append(BatchNormalization(name='L' + str(l) + 'M' + str(m) + 'BN0'))

                layer.append(module)
            weights.append(layer)
        self.weights = np.array(weights)

    def _build_dense_module(self, layer_number, module_number, module_input):
        try:
            module = self.weights[layer_number][module_number]
        except IndexError:
            return module_input

        thread = module_input
        for node in module:
            thread = node(thread)

        return thread

    def _build_conv_module(self, layer_index, module_index, module_input, max_pool=None, batchnorm=True):
        module = None
        try:
            module = self.weights[layer_index][module_index]
        except IndexError:
            return module_input

        thread = module_input
        for node in module:
            thread = node(thread)
            if batchnorm:
                pass
                #thread = BatchNormalization()(thread)

        if max_pool is not None:
            thread = MaxPooling2D((2, 2))(thread)

        return thread

    def build_model_from_path(self, mask=None, optimizer=Adam(), loss='categorical_crossentropy'):
        if mask is None:
            mask = [list(range(self.width))]*self.depth

        inp = Input(self.input_shape)
        thread = inp
        for layer_index, active_modules in enumerate(mask[:-1]):
            outputs = [self._build_conv_module(layer_index, i, thread, max_pool=None) for i in active_modules]
            if len(outputs) == 1:
                thread = outputs[0]
            else:
                thread = add(outputs)

        thread = MaxPooling2D((2, 2))(thread)
        thread = Flatten()(thread)

        outputs = [self._build_dense_module(self.depth-1, i, thread) for i in mask[-1]]

        if len(outputs) == 1:
            thread = outputs[0]
        else:
            thread = add(outputs)

        m = Model(inputs=inp, outputs=thread, name='Path'+str(len(self.optimal_paths)))
        m.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return m

    def _lock_modules_from_training(self, path):
        for layer, layer_modules in enumerate(path):
            for module in layer_modules:
                module = self.weights[layer, module]
                for node in module:
                    node.trainable = False

    def _reinitialize_open_layers_uniformly(self, low=-1, high=1):
        ### NOTE:
        ### During curriculum, data is from the same distribution, and reinitialization of batchnormalization layers
        ### is unnecessary
        for layer in self.weights:
            for module in layer:
                for node in module:
                    if node.trainable:
                        if type(node) is BatchNormalization:
                            node.set_weights([
                                np.ones_like(node.get_weights()[0]),
                                np.zeros_like(node.get_weights()[1]),
                                np.zeros_like(node.get_weights()[2]),
                                np.ones_like(node.get_weights()[3])
                            ])
                        else:
                            node.set_weights([
                                np.random.uniform(low=low, high=high, size=node.get_weights()[0].shape),
                                np.zeros_like(node.get_weights()[1])
                            ])

    def save_new_optimal_path(self, path):
        self.optimal_paths.append(path)
        self._lock_modules_from_training(path)
        self._reinitialize_open_layers_uniformly()

        new_counter = np.zeros_like(self.training_counter)

        for p in self.optimal_paths:
            for layer in range(self.depth):
                for module in p[layer]:
                    new_counter[layer][module] = self.training_counter[layer][module]

        self.training_counter = new_counter

    def print_path(self, path):
        empty_line_size = len(self.weights[0][0])*len('l0m0c0-') - 1

        print('\n'+'-'*30*self.depth)
        for i in range(self.width):
            print(end='     ')
            for j in range(self.depth):
                to_print = ''
                if i in path[j]:
                    for node_nr, node in enumerate(self.weights[j][i]):
                        if node.trainable:
                            to_print += node.name
                        else:
                            to_print += 'x'*len(node.name)
                        if node_nr < len(self.weights[j][i])-1:
                            to_print += '-'
                else:
                    to_print+=' '

                print(to_print.ljust(8*len(self.weights[j][i])), end='   ')
            print()
        print('-'*30*self.depth, '\n')

    def print_module_states(self):
        print('\n'+'-'*15*self.depth)
        for i in range(self.width):
            print(end='     ')
            for j in range(self.depth):
                for node_nr, node in enumerate(self.weights[j][i]):
                    if node.trainable:
                        print(end='-')
                    else:
                        print(end='x')
                print(end='   ')
            print()
        print('-'*15*self.depth, '\n')

    def print_training_counter(self):
        print('\n'+'-' * 15 * self.depth)
        for i in range(self.width):
            print(end='     ')
            for j in range(self.depth):
                if self.training_counter[j][i] == 0:
                    print('-'.ljust(5), end='')
                else:
                    print(str(self.training_counter[j][i]).ljust(5), end='')
            print()
        print('-' * 15 * self.depth, '\n')

    def random_path(self, min=1, max=2):
        if min < 1:
            min = 1

        if self.max_number_of_modules_in_layer is not None:
            max = self.max_number_of_modules_in_layer
        if self.min_number_of_modules_in_layer is not None:
            min = self.min_number_of_modules_in_layer

        path = []
        for _ in range(self.depth):
            contenders = list(range(self.width))
            np.random.shuffle(contenders)
            if min == max:
                path.append(contenders[:min])
            else:
                path.append(contenders[:np.random.randint(low=min, high=max+1)])

        return path

    def mutate_path(self, path, mutation_prob=0.1):
        for layer in path.copy():
            for i in range(len(layer)):
                if np.random.uniform(0, 1) <= mutation_prob:
                    layer[i] += np.random.randint(low=-2, high=2)
                    if layer[i] < 0:
                        layer[i] += self.depth
                    if layer[i] >= self.depth:
                        layer[i] -= self.depth
        for i in range(self.depth):
            path[i] = list(set(path[i]))
            path[i] = sorted(path[i])

        return path

    def increment_training_counter(self, path):
        for layer in range(self.depth):
            for module in path[layer]:
                self.training_counter[layer][module]+=1

    def save_pathnet(self):
        log = {
            'paths': self.optimal_paths,
            'depth': self.depth,
            'width': self.width,
            'in_shape': self.input_shape,
            'out_size': self.output_size,
            'training_counter': self.training_counter,
            'dense_module_config': self.dense_module_config,
            'conv_module_config': self.conv_module_config,
            'conv_module_includes_batchnorm':self.conv_module_includes_batchnorm
            }

        model = self.build_model_from_path()
        model_structure = model.to_json()

        with open('logs/log_dict.pkl', 'wb') as f:
            pickle.dump(log, f)

        with open("logs/pathnet_structure.json", "w") as json_file:
            json_file.write(model_structure)

        model.save_weights("logs/pathnet_weights.h5")

    def load_pathnet(self):
        log = None
        with open('logs/log_dict.pkl', 'rb') as f:
            log = pickle.load(f)

        self.optimal_paths = log['paths']
        self.depth = log['depth']
        self.width = log['width']
        self.input_shape = log['in_shape']
        self.output_size = log['out_size']
        self.training_counter = log['training_counter']
        self.dense_module_config = log['dense_module_config']
        self.conv_module_config = log['conv_module_config']
        self.conv_module_includes_batchnorm = log['conv_module_includes_batchnorm']

        with open('logs/pathnet_structure.json', 'r') as json:
            loaded_model_json = json.read()
        model = model_from_json(loaded_model_json)
        model.load_weights('logs/pathnet_weights.h5')

        self._init_pathnet_weights()
        self.build_model_from_path()

        for l, layer in enumerate(self.weights):
            for m, module in enumerate(layer):
                for n, node in enumerate(module):
                    node_weights = model.get_layer(node.name).get_weights()
                    node.set_weights(node_weights)
                    node.trainable = model.get_layer(node.name).trainable

    def train_path(self, x, y, path=None, epochs=20, batch_size=64, verbose=True, model=None, callback=None):
        if path is None:
            path = self.random_path(max=3)
        if model is None:
            model = self.build_model_from_path(path)

        if callback is None:
            hist = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.2)
        else:
            hist = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.2, callbacks=[callback])

        self.increment_training_counter(path)

        return model, path, hist.history

    def parameters_along_path(self, path):
        model = self.build_model_from_path(path)
        return model.count_params()

    def evaluate_path(self, x, y, path):
        model = self.build_model_from_path(path)
        predictions = model.predict(x)

        hit = 0
        miss = 0
        for p, t in zip(predictions, y):
            if np.argmax(p) == np.argmax(t):
                hit+=1
            else:
                miss+=1

        return hit/(hit+miss)

    def reset_backend(self):
        import keras.backend.tensorflow_backend as tb_backend
        import tensorflow as tf
        tf.reset_default_graph()
        tb_backend._SESSION.close()
        tb_backend._SESSION = None

    def reset_backend_session(self):
        weights = {}

        for l in self.weights:
            for m in l:
                for n in m:
                    weights[n.name] = n.get_weights()

        import keras.backend.tensorflow_backend as tb_backend
        import tensorflow as tf
        tf.reset_default_graph()
        tb_backend._SESSION.close()
        tb_backend._SESSION = None



                    # K.clear_session()
       # import tensorflow as tf
       # K.set_session(tf.Session())

        self._init_pathnet_weights()
        m = self.build_model_from_path()

        for l in self.weights:
            for m in l:
                for n in m:
                    n.set_weights(weights[n.name])  # This takes time!!

