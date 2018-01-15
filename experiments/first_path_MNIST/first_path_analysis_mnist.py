import sys
sys.path.append('../../')
from datetime import datetime as dt
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pickle as pkl
import os
import time as clock
import numpy as np
import random


dir = 'mnist_3layerConv_0.99'
experiments = 31
width = 10
depth = 3
SS_COLOR = '#1f77b4'
PS_COLOR = '#ff7f0e'
RANDOM_COLOR = '#2ca02c'

log = None
with open(dir+'/log.pkl', 'rb') as file:
    log = pkl.load(file)

assert log is not None, dir + ' is not valid log name'
print('NUMBER OF EXPERIMENTS:', len(log['s+s:path1']))

numb_exp = len(log['s+s:path1'])
# Probabilities of index number of overlap between two randomly chosen models
# given width = 10, depth = 3 and uniformly chosen number of active modules between 1 and 3 from
# each layer
overlap_prob = [0.2604032750, 0.3960797996, 0.2457475490, 0.0806327611, 0.0152829566,
                0.0014405546, 0.0001172514, 0.0000045799, 0.0000000945, 0.0000000008]
ssbox = [[], [], [], [], []]
psbox = [[], [], [], [], []]

for ss_avg, ss_mr, ps_avg, ps_mr in zip(log['s+s:avg_training2'], log['s+s:module_reuse'],log['p+s:avg_training2'], log['p+s:module_reuse']):
    ssbox[ss_mr].append(ss_avg)
    psbox[ps_mr].append(ps_avg)


def draw_plot(data, offset, edge_color, fill_color):
    pos = np.arange(len(ssbox))+offset
    bp = ax.boxplot(data, positions=pos, widths=0.2, patch_artist=True, manage_xticks=False)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

fig, ax = plt.subplots()
draw_plot(ssbox, -0.1, SS_COLOR, "white")
draw_plot(psbox, +0.1, PS_COLOR, "white")
plt.xticks(range(4))
plt.ylabel('Avg training')
plt.xlabel('Module reuse')
ss_patch = mpatches.Patch(color=SS_COLOR, label='Search + Search')
ps_patch = mpatches.Patch(color=PS_COLOR, label='Pick + Search')
plt.legend(handles=[ss_patch, ps_patch])
plt.show()


most_reuse = max(log['s+s:module_reuse']+log['p+s:module_reuse'])
bins = np.linspace(0, most_reuse, 16)
layer_sorted_reuse = []
total_reuse = [int(round(x*numb_exp)) for x in overlap_prob[:most_reuse]]
print(total_reuse)
for path1, ss2, ps2 in zip(log['s+s:path1'], log['s+s:path2'], log['p+s:path2']):
    layer = [0, 0, 0]
    for i in range(len(path1)):
        selected = random.choice([ss2[i], ps2[i]])
        random_modules = list(range(width))
        random.shuffle(random_modules)

        for j in random_modules[:len(selected)]:
            if j in path1[i]:
                layer[i]+=1
    layer_sorted_reuse.append(layer)


data = np.vstack([log['s+s:module_reuse'], log['p+s:module_reuse']])#, total_reuse]).T
plt.hist(data, bins, alpha=0.7, label=['Search + Search', 'Pick + Search'])#, 'random module selection'])
plt.legend(loc='upper right')
plt.xlabel('Module reuse')
plt.ylabel('Frequency')
plt.show()

ss = [0, 0, 0]
ps = [0, 0, 0]

ssoverlap = np.zeros([3, 3, 3, 4])
psoverlap = np.zeros([3, 3, 3, 4])

for i in range(len(log['s+s:path1'])):
    ss1 = log['s+s:path1'][i]
    ss2 = log['s+s:path2'][i]
    ps1 = log['p+s:path1'][i]
    ps2 = log['p+s:path2'][i]

    for j in [0, 1, 2]:
        sso = 0
        pso = 0
        for m in ss1[j]:
            if m in ss2[j]:
                sso+=1
        for m in ps1[j]:
            if m in ps2[j]:
                pso+=1
        ss[j]+=sso
        ps[j]+=pso

        ssoverlap[j][len(ss1[j])-1][len(ss2[j])-1][sso] += 1
        psoverlap[j][len(ss1[j])-1][len(ss2[j])-1][pso] += 1

print('PREPARED NUMBER OF EXAMPLES:', i+1)

rs = [0, 0, 0]
for r in layer_sorted_reuse:
    rs[0] += r[0]
    rs[1] += r[1]
    rs[2] += r[2]

plt.scatter([1, 2, 3], ss, label='S+S')
plt.scatter([1, 2, 3], ps, label='P+S')
plt.scatter([1, 2, 3], rs, label='Random')
plt.legend()
plt.xlabel('Layer')
plt.ylabel('Module reuse')
plt.show()
