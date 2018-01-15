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

WRITE_LOG           = True
accuracy_threshold  = 0.50

search_hyper_param = {'batch_size': 16,
                      'training_iterations': 50,
                      'population_size': 64}
dir_name            = 'umbrella_3layerConv_0.975'

data = DataPrep()
data.mnist()

x1, y1, x_test1, y_test1 = data.sample_dataset([5, 6, 7, 8, 9])
x2, y2, x_test2, y_test2 = data.sample_dataset([0, 1, 2, 3, 4])
x3, y3, x_test3, y_test3 = data.sample_dataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

try:
    with open(dir_name+'/log.pkl', 'rb') as file:
        log = pkl.load(file)
except FileNotFoundError:
    log = {'path1': [], 'path2': [], 'path3': [],
           'eval1': [], 'eval2': [], 'eval3': [],
           'reuse12': [], 'reuse13': [], 'reuse23': [],
           'gen1': [], 'gen2': [], 'gen3': [],
           'avg_training1': [], 'avg_training2': [], 'avg_training3': [],
           'base_path': [],
           'base_eval': [],
           'base_gen': [],
           'base_avg_training': []
           }


iteration = len(log['path1'])
while True:
    START = clock.time()
    iteration+=1

    search_hyper_param['generation_limit'] = 500

    print('\n'*3, '\t'*3, 'ITERATION NR', iteration, '\n'*2)
    pn, first_task = PathNet.mnist(output_size=5)
    ps = PathSearch(pn)

    path1, fit1, log1 = ps.tournamet_search(x1, y1, first_task, stop_when_reached=accuracy_threshold, hyperparam=search_hyper_param)
    pn.save_new_optimal_path(path1, first_task)
    pn.reset_backend_session()

    second_task = pn.create_new_task(like_this=first_task)
    path2, fit2, log2 = ps.tournamet_search(x2, y2, second_task, stop_when_reached=accuracy_threshold, hyperparam=search_hyper_param)
    pn.save_new_optimal_path(path2, second_task)

    third_task_config = second_task.get_defining_config()
    third_task_config['output'] = 10
    third_task = pn.create_new_task(config=third_task_config)
    path3, fit3, log3 = ps.tournamet_search(x3, y3, third_task, stop_when_reached=accuracy_threshold, hyperparam=search_hyper_param)
    pn.save_new_optimal_path(path3, third_task)

    eval1 = pn.evaluate_path(x_test1, y_test1, path1, first_task)
    eval2 = pn.evaluate_path(x_test2, y_test2, path2, second_task)
    eval3 = pn.evaluate_path(x_test3, y_test3, path3, third_task)

    _, avg1 = Analytic.training_along_path(path1, pn.training_counter)
    _, avg2 = Analytic.training_along_path(path2, pn.training_counter)
    _, avg3 = Analytic.training_along_path(path3, pn.training_counter)

    gen1 = len(log1['path'])
    gen2 = len(log2['path'])
    gen3 = len(log3['path'])

    reuse = Analytic.three_path_overlap(path1, path2, path3)

    print('\tTask one:', 'Avg training:', '%.1f' % avg1, 'Fitness:', '%.4f' % eval1)
    print('\tTask two:', 'Avg training:', '%.1f' % avg2, 'Fitness:', '%.4f' % eval2)
    print('\tUmbrella:', 'Avg training:', '%.1f' % avg3, 'Fitness:', '%.4f' % eval3)
    print('\tReuse:\n\t\tIn path2 from path1:', reuse[0], '\n\t\tIn path3 from path1:',
          reuse[1], '\n\t\tIn path3 from path2:', reuse[2], '\n\n')

    if WRITE_LOG:
        pn_plotter = PathNetPlotter(pn)
        pn_plotter.plot_paths([path1, path2, path3], filename=dir_name+'/umbr:iter'+str(iteration))
    else:
        pn_plotter = PathNetPlotter(pn)
        pn_plotter.plot_paths([path1, path2, path3])

    STOP = clock.time()
    print('Experiment took', STOP-START, 'seconds')

    pn, base_task = PathNet.mnist(output_size=10)
    ps = PathSearch(pn)

    search_hyper_param['generation_limit'] = 1500
    pathB, fitB, logB = ps.tournamet_search(x3, y3, base_task, stop_when_reached=accuracy_threshold, hyperparam=search_hyper_param)
    pn.save_new_optimal_path(pathB, base_task)
    evalB = pn.evaluate_path(x_test3, y_test3, pathB, base_task)
    pn.reset_backend_session()

    _, avgB = Analytic.training_along_path(pathB, pn.training_counter)
    genB = len(logB['path'])

    log['path1'], log['path2'], log['path3'], log['base_path'] = path1, path2, path3, pathB
    log['eval1'], log['eval2'], log['eval3'], log['base_eval'] = eval1, eval2, eval3, evalB
    log['gen1'],  log['gen2'],  log['gen3'],  log['base_gen']  = gen1, gen2, gen3, genB
    log['avg_training1'] = avg1
    log['avg_training2'] = avg2
    log['avg_training3'] = avg3
    log['base_avg_training'] = avgB
    log['reuse12'] = reuse[0]
    log['reuse13'] = reuse[1]
    log['reuse23'] = reuse[2]

    with open(dir_name + '/log.pkl', 'wb') as f:
        pkl.dump(log, f)