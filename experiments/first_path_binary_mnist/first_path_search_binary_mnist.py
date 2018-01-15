import sys
sys.path.append('../../')
from pathnet_keras import PathNet
from path_search import PathSearch
from analytic import Analytic
from dataprep import DataPrep
from plot_pathnet import PathNetPlotter
from datetime import datetime as dt
from matplotlib import pyplot as plt
import pickle as pkl
import os
import time as clock
import numpy as np


repeates = 2
WRITE_LOG = True
accuracy_threshold = 0.99

data = DataPrep()
data.mnist()
data.add_noise(noise_factor=0.5 , prob=0.15)
x1, y1, x_test1, y_test1 = data.sample_dataset([3, 4])
x2, y2, x_test2, y_test2 = data.sample_dataset([1, 2])

log = {'s+s:path1':[], 's+s:path2':[],
       's+s:eval1':[], 's+s:eval2':[],
       's+s:gen1':[],  's+s:gen2':[],
       's+s:avg_training1':[],
       's+s:avg_training2':[],
       's+s:module_reuse':[],
       'p+s:path1':[], 'p+s:path2':[],
       'p+s:eval1':[], 'p+s:eval2':[],
       'p+s:gen1':[],  'p+s:gen2':[],
       'p+s:avg_training1':[],
       'p+s:avg_training2':[],
       'p+s:module_reuse':[]
       }

if WRITE_LOG:
    date = str(dt.now().date())
    time = str(dt.now().hour) + '-' + str(dt.now().minute)
    now = date+':'+time
    os.system('mkdir '+date + ':' + time)

START = clock.time()
for iteration in range(1, repeates+1):
    print('\n'*3, '\t'*3, 'ITERATION NR', iteration, '\n'*2)
    pn, first_task = PathNet.binary_mnist()
    ps = PathSearch(pn)

    path1, fit1, log1 = ps.tournamet_search(x1, y1, first_task, stop_when_reached=accuracy_threshold)
    evaluation1 = pn.evaluate_path(x_test1, y_test1, path1, first_task)

    pn.save_new_optimal_path(path1, first_task)
    second_task = pn.create_new_task(like_this=first_task)
    path2, fit2, log2 = ps.tournamet_search(x2, y2, second_task, stop_when_reached=accuracy_threshold)
    evaluation2 = pn.evaluate_path(x_test2, y_test2, path2, second_task)

    pn.save_new_optimal_path(path2, second_task)

    pn.reset_backend_session()

    log['s+s:path1'].append(path1)
    log['s+s:path2'].append(path2)
    log['s+s:eval1'].append(evaluation1)
    log['s+s:eval2'].append(evaluation2)
    log['s+s:gen1'].append(len(log1['path']))
    log['s+s:gen2'].append(len(log2['path']))
    _, avg1 = Analytic.training_along_path(path1, pn.training_counter)
    log['s+s:avg_training1'].append(avg1)
    _, avg2 = Analytic.training_along_path(path2, pn.training_counter)
    log['s+s:avg_training2'].append(avg2)
    log['s+s:module_reuse'].append(Analytic.path_overlap(path1, path2))

    print('Task one:', 'Avg training:', '%.1f' % avg1, 'Fitness:', '%.4f' % evaluation1)
    print('Task two:', 'Avg training:', '%.1f' % avg2, 'Fitness:', '%.4f' % evaluation2)
    print('Overlap: ', log['s+s:module_reuse'][-1])

    if WRITE_LOG:
        pn_plotter = PathNetPlotter(pn)
        pn_plotter.plot_paths([path1, path2], filename=now+'/s+s:itr'+str(iteration)+':Reuse'+str(log['s+s:module_reuse'][-1]))
    else:
        pn_plotter = PathNetPlotter(pn)
        pn_plotter.plot_paths([path1, path2])



'''
plt.boxplot([log['s+s:eval1'], log['s+s:eval2']])
plt.show(block=False)
import numpy as np
plt.scatter(log['s+s:module_reuse'], (np.array(log['s+s:eval1']) + np.array(log['s+s:eval2']))/2)
plt.show(block=False)
'''

for iteration in range(1, repeates+1):
    print('\n'*3, '\t'*3, 'ITERATION NR', iteration, '\n'*2)
    pn, first_task = PathNet.binary_mnist()
    ps = PathSearch(pn)

    path1 = log['s+s:path1'][iteration-1]
    model = pn.path2model(path1, first_task)
    training_iterations = log['s+s:gen1'][iteration-1]
    hist = []
    print('Training random path:', path1, 'for', training_iterations,'x 50 mini batches')
    for training_iteration in range(training_iterations):
        for batch_nr in range(50):
            batch = np.random.randint(0, len(x1), 16)
            hist.append(model.train_on_batch(x1[batch], y1[batch])[1])
        pn.increment_training_counter(path1)
    evaluation1 = pn.evaluate_path(x_test1, y_test1, path1, first_task)
    pn.save_new_optimal_path(path1, first_task)

    second_task = pn.create_new_task(like_this=first_task)
    path2, fit2, log2 = ps.tournamet_search(x2, y2, second_task, stop_when_reached=accuracy_threshold)
    evaluation2 = pn.evaluate_path(x_test2, y_test2, path2, second_task)

    pn.save_new_optimal_path(path2, second_task)

    pn.reset_backend_session()

    print('Task one:', 'Avg training:', training_iterations, 'Fitness:', '%.4f' % evaluation1)
    print('Task two:', 'Avg training:', '%.1f' % avg2, 'Fitness:', '%.4f' % evaluation2)
    print('Overlap: ', Analytic.path_overlap(path1, path2))
    print('Overlap in s+s:', log['s+s:module_reuse'][iteration-1])

    log['p+s:path1'].append(path1)
    log['p+s:path2'].append(path2)
    log['p+s:eval1'].append(evaluation1)
    log['p+s:eval2'].append(evaluation2)
    log['p+s:gen1'].append(training_iterations)
    log['p+s:gen2'].append(len(log2['path']))
    _, avg1 = Analytic.training_along_path(path1, pn.training_counter)
    log['p+s:avg_training1'].append(avg1)
    _, avg2 = Analytic.training_along_path(path2, pn.training_counter)
    log['p+s:avg_training2'].append(avg2)
    log['p+s:module_reuse'].append(Analytic.path_overlap(path1, path2))

    if WRITE_LOG:
        pn_plotter = PathNetPlotter(pn)
        pn_plotter.plot_paths([path1, path2], filename=now+'/p+s:itr'+str(iteration)+':Reuse'+str(log['s+s:module_reuse'][-1]))
    else:
        pn_plotter = PathNetPlotter(pn)
        pn_plotter.plot_paths([path1, path2])

print('It took', clock.time()-START)

with open(now + '/log.pkl', 'wb') as f:
    pkl.dump(log, f)