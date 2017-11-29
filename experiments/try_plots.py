from plotcreator import Exp1_plotter
import pickle

log = None
dir = '2017-11-26:13-33'
with open(dir+'/log.pkl', 'rb') as file:
    log = pickle.load(file)

print('LOG WITH', len(log['s+s:path1']), 'EXPERIMENTS')
plotter = Exp1_plotter(log, 10, 3)
plotter.training_boxplot(lock=False, save_file=dir+'_training_boxplot')
plotter.module_reuse_histogram(lock=False, save_file=dir+'_module_reuse_histogram')
plotter.module_reuse_by_layer(lock=True, save_file=dir+'_reuse_by_layer')