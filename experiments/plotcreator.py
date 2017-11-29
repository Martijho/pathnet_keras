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

        self.ss_color = '#1f77b4'
        self.ps_color = '#ff7f0e'
        self.random_color ='#2ca02c'

        self.max_reuse = max(self.log['s+s:module_reuse']+self.log['p+s:module_reuse'])

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

    def module_reuse_histogram(self, save_file=None, lock=True):
        plt.figure('Module reuse')
        plt.title('Module reuse histogram')
        bins = np.linspace(0, max(self.log['s+s:module_reuse'] + self.log['p+s:module_reuse']), 16)
        total_reuse = []
        for path1, ss2, ps2 in zip(self.log['s+s:path1'], self.log['s+s:path2'], self.log['p+s:path2']):
            total = 0
            for i in range(len(path1)):
                selected = random.choice([ss2[i], ps2[i]])
                random_modules = list(range(self.width))
                random.shuffle(random_modules)

                for j in random_modules[:len(selected)]:
                    if j in path1[i]:
                        total += 1
            total_reuse.append(total)

        data = np.vstack([self.log['s+s:module_reuse'], self.log['p+s:module_reuse'], total_reuse]).T
        plt.hist(data, bins, alpha=0.7, label=['Search + Search', 'Pick + Search', 'random module selection'],
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
        ss = [0]*self.max_reuse
        ps = [0]*self.max_reuse

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

        layer_sorted_reuse = []
        for path1, ss2, ps2 in zip(self.log['s+s:path1'], self.log['s+s:path2'], self.log['p+s:path2']):
            layer = [0] * self.max_reuse
            for i in range(len(path1)):
                selected = random.choice([ss2[i], ps2[i]])
                random_modules = list(range(self.width))
                random.shuffle(random_modules)

                for j in random_modules[:len(selected)]:
                    if j in path1[i]:
                        layer[i] += 1
            layer_sorted_reuse.append(layer)

        rs = [0]*self.depth
        for r in layer_sorted_reuse:
            for i in range(len(rs)):
                rs[i] += r[i]

        x = list(range(1, self.depth+1))
        plt.scatter(x, ss, label='S+S', color=self.ss_color)
        plt.scatter(x, ps, label='P+S', color=self.ps_color)
        plt.scatter(x, rs, label='Random', color=self.random_color)
        plt.legend()
        plt.xlabel('Layer')
        plt.ylabel('Module reuse')
        if save_file is not None:
            plt.savefig(save_file+ '.png')
        plt.show(block=lock)
