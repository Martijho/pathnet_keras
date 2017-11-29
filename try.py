from pathnet_keras import PathNet
from dataprep import DataPrep
from analytic import Analytic
from path_search import PathSearch
from plot_pathnet import PathNetPlotter

data = DataPrep()
data.mnist()
x1, y1, x_test1, y_test1 = data.sample_dataset([0, 5])
x2, y2, x_test2, y_test2 = data.sample_dataset([2, 7])



pathnet, task = PathNet.binary_mnist()
pathsearch = PathSearch(pathnet)
analytic = Analytic(pathnet)
pathnet_plotter = PathNetPlotter(pathnet)
paths_to_plot = []

print('\t\t\t\tTASK 1')
optimal_path, _, _ = pathsearch.binary_mnist_tournamet_search(x1, y1, task, stop_when_reached=0.95)
print(optimal_path)
analytic.plot_training_counter(lock=False)
pathnet.save_new_optimal_path(optimal_path, task)

###### TEST OPTIMAL PATH 1 #######
print('Results on task 1')
print(pathnet.evaluate_path(x_test1, y_test1, optimal_path, task))
pathnet.path2model(optimal_path, task).summary()
analytic.show_locked_modules()
paths_to_plot.append(optimal_path)



print('\t\t\t\tTASK 2')
task = pathnet.create_new_task(like_this=task)
optimal_path, _, _ = pathsearch.binary_mnist_tournamet_search(x2, y2, task, stop_when_reached=0.95)
print(optimal_path)
analytic.plot_training_counter()
pathnet.save_new_optimal_path(optimal_path, task)

print('Results on task 2')
print(pathnet.evaluate_path(x_test2, y_test2, optimal_path, task))
pathnet.path2model(optimal_path, task).summary()
pathnet.save_new_optimal_path(optimal_path, task)
paths_to_plot.append(optimal_path)


analytic.show_locked_modules()
analytic.plot_training_counter()
analytic.show_optimal_paths()


pathnet_plotter.plot_paths(paths_to_plot)

