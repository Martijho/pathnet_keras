from keras.datasets import cifar10, cifar100, mnist
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np
import random


class DataPrep:
    def __init__(self):
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None
        self.y_list = None
        self.y_test_list = None

        self.input_shape = None
        self.output_size = None

    def cifar10(self, train_size=None, test_size=None):
        self.input_shape = [32, 32, 3]
        self.output_size = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.y_list = y_train
        self.y_test_list = y_test

        self.y = to_categorical(y_train, self.output_size)
        self.y_test = to_categorical(y_test, self.output_size)

        self.x = x_train.reshape([len(x_train), self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        self.x_test = x_test.reshape([len(x_test), self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        if train_size is not None:
            self.x = self.x[:train_size]
            self.y = self.y[:train_size]

        if test_size is not None:
            self.y_test = self.y_test[:test_size]
            self.x_test = self.x_test[:test_size]

    def cifar100(self, train_size=None, test_size=None):
        self.input_shape = [32, 32, 3]
        self.output_size = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        self.y_list = y_train
        self.y_test_list = y_test

        self.y = to_categorical(y_train, self.output_size)
        self.y_test = to_categorical(y_test, self.output_size)

        self.x = x_train.reshape([len(x_train), self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        self.x_test = x_test.reshape([len(x_test), self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        if train_size is not None:
            self.x = self.x[:train_size]
            self.y = self.y[:train_size]

        if test_size is not None:
            self.y_test = self.y_test[:test_size]
            self.x_test = self.x_test[:test_size]

    def mnist(self, train_size=None, test_size=None):
        self.input_shape = [28, 28, 1]
        self.output_size = 10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.y_list = y_train
        self.y_test_list = y_test

        self.y = to_categorical(y_train, self.output_size)
        self.y_test = to_categorical(y_test, self.output_size)

        self.x = x_train.reshape([len(x_train), self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        self.x_test = x_test.reshape([len(x_test), self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        if train_size is not None:
            self.x = self.x[:train_size]
            self.y = self.y[:train_size]

        if test_size is not None:
            self.y_test = self.y_test[:test_size]
            self.x_test = self.x_test[:test_size]

    def get_class(self, label):
        try:
            training_indecies = list((self.y_list == label)[:,0])
            validation_indecies = list((self.y_test_list == label)[:,0])
        except IndexError:
            training_indecies = self.y_list == label
            validation_indecies = self.y_test_list == label

        return self.x[training_indecies], self.y[training_indecies], \
               self.x_test[validation_indecies], self.y_test[validation_indecies]

    def shuffle_data(self, x, y, x_test, y_test):
        x = np.concatenate(x)
        y = np.concatenate(y)
        x_test = np.concatenate(x_test)
        y_test = np.concatenate(y_test)

        training = list(zip(x, y))
        validation = list(zip(x_test, y_test))

        random.shuffle(training)
        random.shuffle(validation)

        x, y = zip(*training)
        x_test, y_test = zip(*validation)

        return np.array(x), np.array(y), np.array(x_test), np.array(y_test)

    def sample_dataset(self, labels):
        x = []
        y = []
        x_test = []
        y_test = []

        for l in labels:
            xi, yi, x_testi, y_testi = self.get_class(l)
            x.append(xi)
            y.append(yi)
            x_test.append(x_testi)
            y_test.append(y_testi)
        x, y, x_test, y_test = self.shuffle_data(x, y, x_test, y_test)

        y = y[:, ~np.all(y == 0, axis=0)]
        y_test = y_test[:, ~np.all(y_test == 0, axis=0)]

        '''
        if y.shape[1] == 2:
            y = y[:, 0]
            y_test = y_test[:, 0]
        '''
        return x, y, x_test, y_test

    def add_noise(self, noise_factor=0.5, prob=0.35):

        N, wi, he, ch = self.x.shape
        noise  = np.random.rand(N, wi, he, ch)
        self.x[noise < prob] = 0                    # Pepper
        self.x[noise < prob*noise_factor] = 255     # Salt

        N, wi, he, ch = self.x_test.shape
        noise = np.random.rand(N, wi, he, ch)
        self.x_test[noise < prob] = 0                    # Pepper
        self.x_test[noise < prob*noise_factor] = 255     # Salt



if __name__ == "__main__":
    data = DataPrep()
    data.mnist()
    plt.figure('Normal data')
    plt.imshow(data.x[121].reshape([28, 28]), cmap='gray')
    plt.show(block=False)

    data.add_noise(noise_factor=0.5, prob=0.15)
    plt.figure('Noisy data')
    plt.imshow(data.x[121].reshape([28, 28]), cmap='gray')
    plt.show(block=True)
