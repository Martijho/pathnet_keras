from pathnet_keras import PathNet
from dataprep import DataPrep
from path_search import PathSearch
from analytic import Analytic
from matplotlib import pyplot as plt

data = DataPrep()
data.mnist()
x1, y1, x_test, y_test = data.sample_dataset([1, 2])
x2, y2, x_test, y_test = data.sample_dataset([3, 4])



log = []

for i in range(1, 2):
    print('\t\t\tRound', i)

    pathnet, first = PathNet.binary_mnist()
    pathsearch = PathSearch(pathnet)
    analytic = Analytic(pathnet)

    path, fitness, l = pathsearch.binary_mnist_tournamet_search(x1, y1, first, stop_when_reached=0.99)
    l['training_counter'] = pathnet.training_counter
    log.append(l)




for l in log:
    paths   = l['path']
    fit     = l['fitness']
    trcnt   = l['training_counter']
    train   = l['avg_training']

    x = list(range(50))
    avg_a = []
    avg_b = []
    fit_a = []
    fit_b = []
    for generation, path, training in zip(fit, paths, train):
        a = generation[0]
        b = generation[1]

        avg_a.append(sum(a)/len(a))
        avg_b.append(sum(b)/len(b))
        fit_a.append(training[0])
        fit_b.append(training[1])

        plt.plot(x, a)
        plt.plot(x, b)

        for i in range(len(x)): x[i] += 50
    plt.show()
    plt.scatter(fit_a, avg_a)
    plt.scatter(fit_b, avg_b)
    plt.ylabel('Avg. fitness')
    plt.xlabel('Avg. training')
    plt.legend(['A: Training[avg. over modules] over avg fitness for this path', 'B: Training[avg. over modules] over avg fitness for this path'])
    plt.show()
