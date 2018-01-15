import sys
sys.path.append('../')
from datetime import datetime as dt
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pickle as pkl
import os
import time as clock
import numpy as np
import random

class Exp1_plotter:
    def __init__(self, log, width, depth):
        self.log = log
        self.width = width
        self.depth = depth
        self.number_of_experiments = len(log['s+s:path1'])

        self.ss_color = '#1f77b4'
        self.ps_color = '#ff7f0e'
        self.random_color ='#2ca02c'

        self.max_reuse = max(self.log['s+s:module_reuse']+self.log['p+s:module_reuse'])

        # Probabilities of index number of overlap between two randomly chosen models
        # given width = 10, depth = 3 and uniformly chosen number of active modules between 1 and 3 from
        # each layer
        self.overlap_prob = [0.2604032750, 0.3960797996, 0.2457475490, 0.0806327611, 0.0152829566,
                             0.0014405546, 0.0001172514, 0.0000045799, 0.0000000945, 0.0000000008]

        self.module_reuse_prob = [0.6385, 0.32376, 0.03672, 0.00092]

        print('Stored metrics in log: ')
        for k, v in log.items():
            if 's+s' in k:
                print('  >', k[4:])

    def training_boxplot(self, save_file=None, lock=True):
        def draw_plot(data, offset, edge_color, fill_color):
            pos = np.arange(self.max_reuse+1)+offset
            bp = ax.boxplot(data, positions=pos, widths=0.2, patch_artist=True, manage_xticks=False)
            for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bp[element], color=edge_color)
            for patch in bp['boxes']:
                patch.set(facecolor=fill_color)

        ssbox = [[] for _ in range(self.max_reuse+1)]
        psbox = [[] for _ in range(self.max_reuse+1)]

        for ss_avg, ss_mr, ps_avg, ps_mr in zip(self.log['s+s:avg_training2'], self.log['s+s:module_reuse'],
                                                self.log['p+s:avg_training2'], self.log['p+s:module_reuse']):
            ssbox[ss_mr].append(ss_avg)
            psbox[ps_mr].append(ps_avg)


        fig, ax = plt.subplots()
        plt.title('Average training vs Module reuse')
        draw_plot(ssbox, -0.1, self.ss_color, "white")
        draw_plot(psbox, +0.1, self.ps_color, "white")
        plt.xticks(range(self.max_reuse+1))
        plt.ylabel('Avg training')
        plt.xlabel('Module reuse')
        ss_patch = mpatches.Patch(color=self.ss_color, label='Search + Search')
        ps_patch = mpatches.Patch(color=self.ps_color, label='Pick + Search')
        plt.legend(handles=[ss_patch, ps_patch])
        if save_file is not None:
            plt.savefig(save_file+ '.png')
        plt.show(block=lock)

        for i in range(len(ssbox)):
            ssbox[i] = sum(ssbox[i]) / len(ssbox[i])
            psbox[i] = sum(psbox[i]) / len(psbox[i])

        plt.plot(ssbox, color=self.ss_color)
        plt.plot(psbox, color=self.ps_color)

    def module_reuse_histogram(self, save_file=None, lock=True):
        plt.figure('Module reuse')
        plt.title('Module reuse histogram')

        random_selection_overlap = []
        for i, P in enumerate(self.overlap_prob):
            random_selection_overlap+=int(round(self.number_of_experiments*P))*[i]

        while len(random_selection_overlap) < len(self.log['s+s:module_reuse']):
            random_selection_overlap.append(1)


        data = np.vstack([self.log['s+s:module_reuse'], self.log['p+s:module_reuse'], random_selection_overlap]).T

        bins = np.linspace(0, max(random_selection_overlap + self.log['p+s:module_reuse'] + self.log['s+s:module_reuse']), 16)

        plt.hist(data, bins, alpha=0.7, label=['Search + Search', 'Pick + Search', 'Overlap in random module selecton'],
                 color=[self.ss_color, self.ps_color, self.random_color])

        plt.legend(loc='upper right')
        plt.xlabel('Module reuse')
        plt.ylabel('Frequency')
        if save_file is not None:
            plt.savefig(save_file+ '.png')
        plt.show(block=lock)

    def module_reuse_by_layer(self, save_file=None, lock=True):
        plt.figure('Module reuse for each layer')
        plt.title('Module reuse for each layer')
        ss = [0]*self.depth
        ps = [0]*self.depth

        for i in range(len(self.log['s+s:path1'])):
            ss1 = self.log['s+s:path1'][i]
            ss2 = self.log['s+s:path2'][i]
            ps1 = self.log['p+s:path1'][i]
            ps2 = self.log['p+s:path2'][i]

            for j in range(self.depth):
                sso = 0
                pso = 0
                for m in ss1[j]:
                    if m in ss2[j]:
                        sso += 1
                for m in ps1[j]:
                    if m in ps2[j]:
                        pso += 1
                ss[j] += sso
                ps[j] += pso



        rs = 0
        for i, P in enumerate(self.module_reuse_prob):
            rs += int(round(self.number_of_experiments*P))*i
        rs = [rs]*self.depth

        x = list(range(1, self.depth+1))

        plt.scatter(x, ss, label='S+S', color=self.ss_color)
        plt.scatter(x, ps, label='P+S', color=self.ps_color)
        plt.plot(x, rs, label='Random', color=self.random_color)
        plt.legend()
        plt.xlabel('Layer')
        plt.ylabel('Module reuse')
        if save_file is not None:
            plt.savefig(save_file+ '.png')
        plt.show(block=lock)

    def evaluation_vs_training(self, save_file=None, lock=True):
        #plt.figure('Model evaluation by avg training')
        #plt.title('Evaluation as function of training')
        max_training = max(self.log['s+s:avg_training1']+self.log['s+s:avg_training2']+self.log['p+s:avg_training1']+self.log['p+s:avg_training2'])

        f, axarr = plt.subplots(2, 2)

        self._eval_vs_training_subploter(axarr[0, 0], self.log['s+s:avg_training1'], self.log['s+s:eval1'],
                                         self.ss_color, 'o', 'S+S: Task 1')
        self._eval_vs_training_subploter(axarr[0, 1], self.log['s+s:avg_training2'], self.log['s+s:eval2'],
                                         self.ss_color, 'x', 'S+S: Task 2')
        self._eval_vs_training_subploter(axarr[1, 0], self.log['p+s:avg_training1'], self.log['p+s:eval1'],
                                         self.ps_color, 'o', 'P+S: Task 1')
        self._eval_vs_training_subploter(axarr[1, 1], self.log['p+s:avg_training2'], self.log['p+s:eval2'],
                                         self.ps_color, 'x', 'P+S: Task 2')


        #plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
        #plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)



        #plt.scatter(self.log['s+s:avg_training1'], self.log['s+s:eval1'], color=self.ss_color, marker='o')
        #plt.scatter(self.log['s+s:avg_training2'], self.log['s+s:eval2'], color=self.ss_color, marker='x')
        #plt.scatter(self.log['p+s:avg_training1'], self.log['p+s:eval1'], color=self.ps_color, marker='o')
        #plt.scatter(self.log['p+s:avg_training2'], self.log['p+s:eval2'], color=self.ps_color, marker='x')

        #plt.legend(['s+s: Task 1', 's+s: Task 2', 'p+s: Task 1', 'p+s: Task 2'])
        #plt.xlabel('average training')
        #plt.ylabel('evaluation accuracy')

        if save_file is not None:
            plt.savefig(save_file+ '.png')
        plt.show(block=lock)


    def _eval_vs_training_subploter(self, subplot, training, eval, color, marker, title):
        training = np.array(training)
        eval = np.array(eval)

        none_outliers = np.array(eval) > 0.85
        training = training[none_outliers]
        eval = eval[none_outliers]

        subplot.scatter(training, eval, color=color, marker=marker)
        subplot.set_title(title)
        fit1 = np.polyfit(training, eval, 1)

        x = np.linspace(0, max(training), 1000)
        subplot.plot(x, fit1[0] * x + fit1[1], color='red')
        subplot.legend([str(fit1[0])+'X + ' + str(fit1[1])])