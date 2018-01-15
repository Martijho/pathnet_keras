from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers.merge import add
from keras.optimizers import Adagrad, Adam, RMSprop, SGD
import numpy as np
import os
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Layer:
    def __init__(self, width, name_prefix, config):
        self._width = width
        self._name_prefix = name_prefix
        self._config = config

        self._modules = []
        self._layers = {}
        self._saved = {}

        self._saved_weights = {}
        self._saved_trainable_state = {}

        self._initialized_weights = {}

    def get_module(self, nr):
        return self._modules[nr]

    def lock_modules(self, selection):
        for module in selection:
            for node in self._modules[module]:
                node.trainable = False

    def is_module_trainable(self, module):
        return self._modules[module][0].trainable

    def save_layer_weights(self):
        self._saved_weights = {}
        self._saved_trainable_state = {}
        for name, layer in self._layers.items():
            self._saved_weights[name] = layer.get_weights()
            self._saved_trainable_state[name] = layer.trainable

    def load_layer_weights(self):
        for module in self._modules:
            for node in module:
                node.set_weights(self._saved_weights[node.name])
                node.trainable = self._saved_trainable_state[node.name]

    def save_initialized_weights(self):
        for name, layer in self._layers.items():
            self._initialized_weights[name] = copy.deepcopy(layer.get_weights())

    def reinitialize_if_open(self):
        for name, layer in self._layers.items():
            if layer.trainable:
                layer.set_weights(self._initialized_weights[name])

    def load_layer_log(self, log):
        for i in range(len(self._modules)):
            train_state = log['trainable'][i]
            for name, weight in zip(log['names'][i], log['weights'][i]):
                self._layers[name].set_weights(weight)
                self._layers[name].trainable = train_state

    @staticmethod
    def initialize_whole_network(layers, input_size):
        inp = Input(input_size)
        thread = inp
        for layer in layers:
            thread = layer.add_layer_selection(list(range(len(layer._modules))), thread)

        output = Dense(10, activation='softmax', name='unique_layer')(thread)
        model = Model(inputs=inp, outputs=output)
        model.compile(optimizer=SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

class ConvLayer(Layer):
    def __init__(self, width, name_prefix, config, maxpool=False):
        super().__init__(width, name_prefix, config)

        self._maxpool = maxpool

        self._init_layer()

        self._saved_weights = {}
        self._saved_trainable_state = {}

    def _init_layer(self):
        self._modules = []
        self._layers = {}

        for m in range(self._width):
            prefix = self._name_prefix + 'M'+ str(m)

            module = []
            for i, node_config in enumerate(self._config):
                name = prefix + 'C'+str(i)
                node = Conv2D(node_config['channels'], node_config['kernel'], strides=node_config['stride'],
                              activation=node_config['activation'], name=name)
                module.append(node)
                self._layers[name] = node

            name = prefix + 'BN'
            node = BatchNormalization(name=name)
            module.append(node)
            self._layers[name] = node

            if self._maxpool:
                name = prefix +'MP'
                node = MaxPooling2D((2,2), name=name)
                module.append(node)
                self._layers[name] = node

            self._modules.append(module)


    def add_layer_selection(self, active_modules, thread):

        outputs = []
        for m in active_modules:
            module_thread = thread
            for node in self._modules[m]: module_thread = node(module_thread)

            outputs.append(module_thread)

        if len(outputs) == 1: thread = outputs[0]
        else: thread = add(outputs)

        return thread

    def get_layer_log(self):
        weights = []
        names = []
        trainable = []
        for module in self._modules:
            module_weights = []
            module_names = []
            for node in module:
                module_weights.append(node.get_weights())
                module_names.append(node.name)

            weights.append(module_weights)
            names.append(module_names)
            trainable.append(module[0].trainable)

        return  {
                'layer_type':'conv',
                'weights':weights,
                'names':names,
                'trainable':trainable,
                'maxpool':self._maxpool,
                'width':self._width,
                'prefix':self._name_prefix,
                'config':self._config
                }

    @staticmethod
    def build_from_log(log):
        ConvLayer(log['width'], log['prefix'], log['config'], log['maxpool'])

class DenseLayer(Layer):
    def __init__(self, width, name_prefix, config, flatten=False):
        super().__init__(width, name_prefix, config)
        self._flatten_first = flatten

        self._init_layer()

        self._saved_weights = {}
        self._saved_trainable_state = {}

    def _init_layer(self):
        self._modules = []
        self._layers = {}
        for m in range(self._width):
            prefix = self._name_prefix + 'M' + str(m)

            module = []
            for i, node_config in enumerate(self._config):
                name = prefix + 'D' + str(i)
                node = Dense(node_config['out'], activation=node_config['activation'], name=name)
                module.append(node)
                self._layers[name] = node

            self._modules.append(module)

    def add_layer_selection(self, active_modules, thread):

        if self._flatten_first:
            thread = Flatten()(thread)

        outputs = []
        for m in active_modules:
            module_thread = thread

            for node in self._modules[m]: module_thread = node(module_thread)

            outputs.append(module_thread)

        if len(outputs) == 1: thread = outputs[0]
        else: thread = add(outputs)

        return thread

    def reinitialize_if_open(self, from_initialized_state=True):
        for name, layer in self._layers.items():
            if layer.trainable:
                if from_initialized_state:
                    layer.set_weights(self._initialized_weights[name])
                else:
                    layer.set_weights([
                            np.random.normal(0, 0.5, size=layer.get_weights()[0].shape),
                            np.zeros_like(layer.get_weights()[1])
                        ])

    def get_layer_log(self):
        weights = []
        names = []
        trainable = []
        for module in self._modules:
            module_weights = []
            module_names = []
            for node in module:
                module_weights.append(node.get_weights())
                module_names.append(node.name)

            weights.append(module_weights)
            names.append(module_names)
            trainable.append(module[0].trainable)

        return  {
                'layer_type':'dense',
                'weights':weights,
                'names':names,
                'trainable':trainable,
                'flatten':self._flatten_first,
                'width':self._width,
                'prefix':self._name_prefix,
                'config':self._config
                }

    @staticmethod
    def build_from_log(log):
        return DenseLayer(log['width'], log['prefix'], log['config'], log['flatten'])



class TaskContainer:
    def __init__(self, input_shape, output_size, flatten_first, name='Unique_layer', optimizer=SGD, lr=0.0001, loss='binary_crossentropy'):
        self.flatten_first = flatten_first
        self.input_shape = input_shape
        self.output_size = output_size
        self.optimal_path = None
        self.name = name
        self.optimizer = optimizer
        self.learningrate = lr
        self.loss = loss

        self.layer = Dense(output_size, activation='softmax', name=name)
        self._pathnet_output_size = None

    def save_layer_weights(self):
        self._saved_weights = self.layer.get_weights()

        if self._pathnet_output_size is None:
            self._pathnet_output_size = self.layer.input_shape

    def load_layer_weights(self):
        self.layer = Dense(self.output_size, activation='softmax', name=self.name)

        inp = Input(self._pathnet_output_size)
        out = self.layer(inp)

        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer=self.optimizer(self.learningrate), loss=self.loss)

        self.layer.set_weights(self._saved_weights)

    def create_task_like_this(self):
        task = TaskContainer(self.input_shape, self.output_size, self.flatten_first,
                             self.name, self.optimizer, self.learningrate, self.loss)
        return task

    def add_unique_layer(self, thread):
        if self.flatten_first:
            thread = Flatten()(thread)
        return self.layer(thread)

    def get_task_log(self):
        return  {
                'input_shape':self.input_shape,
                'output_size':self.output_size,
                'optimal_path':self.optimal_path,
                'name':self.name,
                'optimizer':self.optimizer,
                'lr':self.learningrate,
                'loss':self.loss,
                'layer_weights':self.layer.get_weights(),
                }

    def get_defining_config(self):
        return {'input':    self.input_shape,
                'output':   self.output_size,
                'name':     self.name,
                'optim':    self.optimizer,
                'loss':     self.loss,
                'lr':       self.learningrate,
                'flatten_first': self.flatten_first}
    @staticmethod
    def build_from_log(log):
        tc = TaskContainer(log['input_shape'], log['output_size'], log['name'], log['optimizer'], log['lr'], log['loss'])
        tc.optimal_path = log['optimal_path']
        return tc