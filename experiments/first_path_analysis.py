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

def prob(o, n, a1, a2):
    if a2 == 1:
        if o == 0:
            return 1-(a1/n)
        elif o == 1:
            return a1/n
    elif a2 == 2:
        if o == 0:
            return ((1-a1)**2)/(n*(n-1))
        elif o == 1:
            return (2*a1*(n-a1))/(n*(n-1))
        elif o == 2:
            return (a1*(a1-1))/(n*(n-1))
    elif a2 == 3:
        if o == 0:
            return ((n-a1)*(-a1+n-1)*(-a1+n-2))/(n*(n-1)*(n-2))
        elif o == 1:
            return (3*a1*(a1-n)*(a1-n+1))/(n*(n-1)*(n-2))
        elif o == 2:
            return (a1*(a1-1)*(3*n-3*a1-1))/(n*(n-1)*(n-2))
        elif o == 3:
            return (a1*(a1-1)*(a1-2))/(n*(n-1)*(n-2))

dir = '2017-11-26:13-33'
experiments = 600
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

ssbox = [[], [], [], []]
psbox = [[], [], [], []]

for ss_avg, ss_mr, ps_avg, ps_mr in zip(log['s+s:avg_training2'], log['s+s:module_reuse'],log['p+s:avg_training2'], log['p+s:module_reuse']):
    ssbox[ss_mr].append(ss_avg)
    psbox[ps_mr].append(ps_avg)


def draw_plot(data, offset, edge_color, fill_color):
    pos = np.arange(4)+offset
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



bins = np.linspace(0, max(log['s+s:module_reuse']+log['p+s:module_reuse']), 16)
layer_sorted_reuse = []
total_reuse = []
for path1, ss2, ps2 in zip(log['s+s:path1'], log['s+s:path2'], log['p+s:path2']):
    total = 0
    layer = [0, 0, 0]
    for i in range(len(path1)):
        selected = random.choice([ss2[i], ps2[i]])
        random_modules = list(range(width))
        random.shuffle(random_modules)

        for j in random_modules[:len(selected)]:
            if j in path1[i]:
                total+=1
                layer[i]+=1
    layer_sorted_reuse.append(layer)
    total_reuse.append(total)

data = np.vstack([log['s+s:module_reuse'], log['p+s:module_reuse'], total_reuse]).T
plt.hist(data, bins, alpha=0.7, label=['Search + Search', 'Pick + Search', 'random module selection'])
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

'''
random_results = [[[0.9, 0.1, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.7, 0.3, 0.0, 0.0]],
                  [[0.8, 0.2, 0.0, 0.0], [0.622, 0.356, 0.022, 0.0], [0.467, 0.466, 0.067, 0.0]],
                  [[0.7, 0.3, 0.0, 0.0], [0.467, 0.466, 0.067, 0.0], [0.292, 0.525, 0.175, 0.0]]]

f, axarr = plt.subplots(3, 3)

for a1 in range(3):#[1, 2, 3]:
    for a2 in range(3):#[1, 2, 3]:
        bins = np.arange(4)-0.5
        ss = ssoverlap[0][a1][a2][:]
        print(ss)
        input()
        ps = psoverlap[0][a1][a2][:]*100 / sum(psoverlap[0][a1][a2][:])

        axarr[a1, a2].plot(ss)
        axarr[a1, a2].set_ylim([0, 610])
        #axarr[a1, a2].hist([0, 1, 2, 3], bins, alpha=0.7,  weights=ps, color='blue', label='p+s')
        #axarr[a1, a2].xlabel('# overlap')
        #axarr[a1, a2].ylabel('Frequency')
        axarr[a1, a2].line(y=random_results[a1][a2][0] * 600, xmin=0, xmax=0.5, color='grey')
        axarr[a1, a2].hline(y=random_results[a1][a2][1] * 600, xmin=1, xmax=1.5, color='red')
        axarr[a1, a2].hline(y=random_results[a1][a2][2] * 600, xmin=2, xmax=2.5, color='grey')
        axarr[a1, a2].hline(y=random_results[a1][a2][3] * 600, xmin=3, xmax=3.5, color='red')


plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.show()




# an probability measure of module reuse vs observed module reuse
'''