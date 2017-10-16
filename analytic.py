from matplotlib import pyplot as plt
from pathnet_keras import PathNet
import pickle as pkl

import numpy as np

class Analytic:
    def __init__(self, pathnet):
        self.pathnet = pathnet

    def plot_training_counter(self, lock=True):
        network = np.array(self.pathnet.training_counter).transpose()

        p = plt.imshow(network, cmap='hot', interpolation='nearest', vmin=0)
        plt.colorbar()
        p.axes.get_xaxis().set_visible(False)
        p.axes.get_yaxis().set_visible(False)
        plt.show(block=lock)

    def process_evolution_run(self, history):
        history = self.history_list_to_dict(history)
        self.plot_history(history)
        # TODO
        # save history to file?
        # load history from file?

    def plot_history(self, hist, lock=True):
        plt.figure()
        plt.subplot(2, 2, 1)
        self.plot_history_subplot(hist['val_acc'])
        plt.ylabel('Validation Accuracy')

        plt.subplot(2, 2, 2)
        self.plot_history_subplot(hist['val_loss'])
        plt.ylabel('Validation Loss')

        plt.subplot(2, 2, 3)
        self.plot_history_subplot(hist['acc'])
        plt.ylabel('Training Accuracy')

        plt.subplot(2, 2, 4)
        self.plot_history_subplot(hist['loss'])
        plt.ylabel('Training Loss')

        plt.show(block=lock)

    def plot_history_subplot(self, hist):
        winners = [0]
        loosers = [0]
        average = [0]

        for gen in hist:
            winners.append(max(gen))
            loosers.append(min(gen))
            average.append(sum(gen)/len(gen))

        plt.boxplot(hist)
        #plt.plot(winners)
        #plt.plot(loosers)
        #plt.plot(average)
        #plt.legend(['Boxplot', 'Highest', 'Lowest', 'Average'])

    def history_list_to_dict(self, hist):
        val_acc = []
        val_loss = []
        acc = []
        loss = []
        for gen in hist:
            g_acc = []
            g_loss = []
            g_val_acc = []
            g_val_loss = []
            for eval in gen:
                g_val_acc.append(eval['val_acc'][-1])
                g_val_loss.append(eval['val_loss'][-1])
                g_acc.append(eval['acc'][-1])
                g_loss.append(eval['loss'][-1])
            val_acc.append(g_val_acc)
            val_loss.append(g_val_loss)
            loss.append(g_loss)
            acc.append(g_acc)

        return {'loss':loss, 'acc':acc, 'val_loss':val_loss, 'val_acc':val_acc}

    def show_optimal_paths(self):
        for i, p in enumerate(self.pathnet.optimal_paths):
            print('='*20, 'Task nr'+str(i+1), '='*20)
            print('Path:')
            print(p)
            print('Training counter:')

            modules_training_log = []
            total = 0
            number_of_modules_in_path = 0
            for layer in range(self.pathnet.depth):
                l = []
                for module in p[layer]:
                    l.append(self.pathnet.training_counter[layer][module])
                    total+=self.pathnet.training_counter[layer][module]
                    number_of_modules_in_path+=1
                modules_training_log.append(l)

            print(modules_training_log)
            print('Average epochs trained on each module:', total/number_of_modules_in_path)
            print('\n')
