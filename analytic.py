from matplotlib import pyplot as plt
#from pathnet_keras import PathNet
from datetime import datetime
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

        now = datetime.now()
        with open('logs/History - EA_search/evolutionary_run_'+str(now)+'.pkl', 'wb') as f:
            pkl.dump(history, f)

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

    def training_along_path(self, p):
        modules_training_log = []
        total = 0
        number_of_modules_in_path = 0
        for layer in range(self.pathnet.depth):
            l = []
            for module in p[layer]:
                l.append(self.pathnet.training_counter[layer][module])
                total += self.pathnet.training_counter[layer][module]
                number_of_modules_in_path += 1
            modules_training_log.append(l)
        return modules_training_log, total/number_of_modules_in_path

    def show_optimal_paths(self):
        for i, t in enumerate(self.pathnet._tasks):
            p = t.optimal_path
            if p is None: continue

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

    def _is_module_trainable(self, layer, module):
        prefix = 'L'+str(layer)+'M'+str(module)

        for name, layer in self.pathnet._saved_layers.items():
            if prefix in name:
                return layer.trainable
    def _module_size(self, layer, module):
        prefix = 'L'+str(layer)+'M'+str(module)
        counter = 0
        for name, layer in self.pathnet._saved_layers.items():
            if prefix in name:
                counter+=1
        return counter

    def show_locked_modules(self):
        pn = self.pathnet
        print('='*20, 'Locked Modules', '='*20)
        for m in range(pn.width):
            print(end='\t')
            for l in range(pn.depth):
                if self._is_module_trainable(l, m):
                    print('-'*self._module_size(l, m), end='   ')
                else:
                    print('X'*self._module_size(l, m), end='   ')
            print()
        print('='*56, end='\n\n')

    def print_training_counter(self):
        pn = self.pathnet
        print('='*19, 'Training counter', '='*19)
        for i in range(self.pathnet.width):
            print(end='\t')
            for j in range(self.pathnet.depth):
                if self.pathnet.training_counter[j][i] == 0:
                    print('-'.ljust(5), end='')
                else:
                    print(str(self.pathnet.training_counter[j][i]).ljust(5), end='')
            print()
        print('='*56, end ='\n\n')

    def parameters_along_path(self, path):
        model = self.pathnet.path2model(path)
        return model.count_params()

    def load_log(self):
        with open('logs/History - EA_search/kept/evolutionary_run_2017-10-20.pkl', 'rb') as f:
            log = pkl.load(f)
        for k, v in log.items():
            print(k)

        self.plot_history(log)

if __name__ == "__main__":
    Analytic(None).load_log()