from pathnet_keras import PathNet
import numpy as np
import random
import time

class PathSearch:
    def __init__(self, pathnet):
        self.pathnet = pathnet
        self.time_log = []

    def tournament_search(self, x, y, population_size=10, seasons=5):
        batch_size = 512
        epochs = 1
        pn = self.pathnet
        population = [pn.random_path(max=3) for _ in range(population_size)]
        models = [pn.build_model_from_path(path) for path in population]

        fitness_history = []

        for season in range(1, seasons + 1):
            print(' ' * 20, 'Season ', season, '/', seasons)
            print('Evaluate fitness')

            fitness = [model.fit(x, y, epochs=epochs, validation_split=0.2,
                                 verbose=True, batch_size=batch_size).history['val_acc'][0] for model in models]
            fitness_history.append(sum(fitness) / population_size)
            for path in population:
                pn.increment_training_counter(path)

            for acc, path in zip(fitness, population):
                print(path.__str__().ljust(35), 'Fit:', acc)

            draft = list(range(population_size))
            random.shuffle(draft)
            print('Select, transfer and mutate')
            while len(draft) > 1:
                best = draft.pop()
                worst = draft.pop()
                if fitness[worst] > fitness[best]:
                    tmp = best
                    best = worst
                    worst = tmp

                models[worst] = None
                population[worst] = pn.mutate_path(population[best].copy(),
                                                   mutation_prob=population_size / (pn.depth * pn.width))
                population[best] = pn.mutate_path(population[best].copy(),
                                                  mutation_prob=population_size / (pn.depth * pn.width))

            for i, path in enumerate(population):
                models[i] = pn.build_model_from_path(path)

            pn.print_training_counter()
            best = np.argmax(np.array(fitness))

        return population[best], fitness_history

    def serial_tournament_search(self, x, y, val_x, val_y, max_modules_pr_layer=3, epochs_pr_evaluation=1,
                                 evolutions=20, batch_size=512, verbose=True):
        fitness_history = []
        champion = None
        pn = self.pathnet
        champion_path = pn.random_path(max=max_modules_pr_layer)
        challenger_path = pn.random_path(max=max_modules_pr_layer)

        if verbose:
            print('='*10, 'Serial pathway tournament search', '='*10, end='\n\n')

        for i in range(evolutions):
            if verbose:
                print(' '*15, 'ROUND', i)
                print('Champion:  ', champion_path)
                print('Challenger:', challenger_path, end='\n\n')

            champion = pn.build_model_from_path(champion_path)
            challenger = pn.build_model_from_path(challenger_path)

            champion_fitness = 1 * champion.fit(x, y, epochs=epochs_pr_evaluation, validation_split=0.2,
                                                 verbose=True, batch_size=batch_size).history['val_acc'][0]
            challenger_fitness = 1 * challenger.fit(x, y, epochs=epochs_pr_evaluation, validation_split=0.2,
                                                     verbose=True, batch_size=batch_size).history['val_acc'][0]

            pn.increment_training_counter(challenger_path)
            pn.increment_training_counter(champion_path)

            del challenger
            del champion

            if verbose:
                print('\nChampion fitness:', champion_fitness,'\nChallenger fitness:  ', challenger_fitness)

            if challenger_fitness > champion_fitness:
                if verbose:
                    print('Challenger wins!')
                champion_path = challenger_path
                fitness_history.append(challenger_fitness)
            else:
                fitness_history.append(champion_fitness)
                if verbose:
                 print('Champion wins!')

            pn.print_training_counter()

            if i != evolutions-1:
                champion_path = pn.mutate_path(champion_path, mutation_prob=1/(max_modules_pr_layer*pn.depth))
                challenger_path = pn.random_path(max=max_modules_pr_layer)

        return champion_path, fitness_history

    def binary_mnist_tournamet_search(self, x, y, task_nr=0):
        batch_size = 16
        training_iterations = 50
        population = [self.pathnet.random_path(min=1, max=3) for _ in range(64)]
        generation = 0

        while True:

            path_a = random.choice(population)
            path_b = random.choice(population)
            if path_a == path_b: continue

            generation += 1

            print('='*15, 'Generation', generation, '='*15)
            print(path_a, '\tVs.', path_b, sep='\n')

            a = self.pathnet.path2model(path_a, task_nr=task_nr)
            b = self.pathnet.path2model(path_b, task_nr=task_nr)
            fit_a = 0
            fit_b = 0
            for batch_nr in range(training_iterations):
                batch = np.random.randint(0, len(x), batch_size)
                batch_x = x[batch]
                batch_y = y[batch]
                fit_a = a.train_on_batch(batch_x, batch_y)[1]
                fit_b = b.train_on_batch(batch_x, batch_y)[1]

            self.pathnet.increment_training_counter(path_a)
            self.pathnet.increment_training_counter(path_b)

            fit_a /= training_iterations
            fit_b /= training_iterations
            print(path_a, '\t=>', fit_a)
            print(path_b, '\t=>', fit_b)

            if fit_a > fit_b:
                population[population.index(path_b)] = self.mutate_path(path_a, 1/9)
                if fit_a > 0.999:
                    return path_a, fit_a


    def evolutionary_search(self, x, y, population_size=4, generations=2):
        batch_size = 256
        epochs = 1
        max_modules_pr_layer = 3
        history = []
        population = [self.pathnet.random_path(max=max_modules_pr_layer) for _ in range(population_size)]

        best_path_found = None
        best_fitness = -1

        for generation in range(1, generations+1):
            print('='*15, 'Generation ', generation, '/', generations, '='*15)

            print(' ' * 5, '--- Evaluating fitness of', population_size, 'paths ---')
            fitness, hist = self.evaluate(population, x, y, epochs, batch_size)
            history.append(hist)

            population, fitness = self.sort_generation_by_fitness(population, fitness)

            if best_fitness < fitness[0]:
                best_fitness = fitness[0]
                best_path_found = population[0]

            if generation == generations:
                break

            print(' ' * 5, '--- Selection ---')
            selected, fitness_of_selected = self.selection(population, fitness)

            for f, i in zip(fitness, population):
                print(str(i).ljust(50), str(round(f, 5)).ljust(10), end='')
                if f in fitness_of_selected:
                    print('*')
                else:
                    print()
            print()

            print(' ' * 5, '--- Recombination ---')
            new_population = self.crossover(selected)
            for s in new_population:
                print(s)

            print(' ' * 5, '--- Mutation ---')
            new_population = self.mutate(new_population)
            print('\n\n')

            population = selected + new_population

        return population[0], history

    def evaluate(self, population, x, y, epochs, batch_size):
        fitness = []
        history = []
        for path in population:
            t = time.time()

            model = self.pathnet.path2model(path)
            hist = model.fit(x, y, epochs=epochs, validation_split=0.2, verbose=True, batch_size=batch_size)

            self.time_log.append(time.time()-t / model.count_params())

            self.pathnet.increment_training_counter(path)
            history.append(hist.history)
            fitness.append(hist.history['val_acc'][0])

        return fitness, history

    def select_one_index(self, fitness):
        value = random.random() * sum(fitness)
        for i in range(len(fitness)):
            value -= fitness[i]
            if value <= 0:
                return i

    def selection(self, population, fitness):
        population_size = len(population)
        fit = []
        pop = []
        for f, i in sorted(zip(fitness, population)):
            fit.append(f)
            pop.append(i)

        survived = []
        sur_fit = []

        survived.append(pop.pop())
        sur_fit.append(fit.pop())
        del pop[0]
        del fit[0]

        for _ in range(int(len(population) / 2)-1):
            i = self.select_one_index(fit)
            len_1 = len(pop)
            survived.append(pop[i])
            sur_fit.append(fit[i])

            del pop[i]
            del fit[i]
            len_2 = len(pop)

            assert len_1-1 == len_2, 'Selection(EA): removes too much from population-list. Remove "del pop[i]'

        assert len(survived) == population_size/2, 'Selection(EA): wrong number of survived genotypes'

        return survived, sur_fit

    def combine(self, a, b):
        offspring = []
        for layer_number in range(len(a)):
            layer = []
            for m in a[layer_number]:           # copy duplicate modules
                if m in b[layer_number]:
                    layer.append(m)

            layer_size = (len(a[layer_number]) + len(b[layer_number])) / 2  # Size of layer is mean of parents
            if layer_size - int(layer_size) > 0:                            # if sum is odd, randomly favour one parent
                if random.choice([True, False]):
                    layer_size += 0.5

            layer_size = int(layer_size)

            while len(layer) < layer_size:
                layer.append(random.choice(a[layer_number] + b[layer_number]))
                layer = list(set(layer))

            offspring.append(layer)

        return offspring

    def crossover(self, population):
        old_population_length = len(population)
        new_population = []

        for father in range(0, len(population) - 1, 2):
            mother = father + 1
            new_population.append(self.combine(population[father], population[mother]))

        for father in range(len(population)):
            if len(new_population) == len(population):
                break
            mother = father + int(len(population) / 2)
            if mother >= len(population):
                mother -= len(population)
            new_population.append(self.combine(population[father], population[mother]))

        assert len(new_population) == old_population_length, 'Crossover(EA): wrong number of children' \
                                                             ''
        return new_population

    def mutate(self, population):
        mutated = []
        for p in population:
            N = max([len(x) for x in p])
            L = self.pathnet.depth
            mutated.append(self.mutate_path(p, mutation_prob=1/(N*L)))
        return mutated

    def mutate_path(self, path, mutation_prob=0.1):
        for layer in path.copy():
            for i in range(len(layer)):
                if np.random.uniform(0, 1) <= mutation_prob:
                    layer[i] += np.random.randint(low=-2, high=2)
                    if layer[i] < 0:
                        layer[i] += self.pathnet.depth
                    if layer[i] >= self.pathnet.depth:
                        layer[i] -= self.pathnet.depth
        for i in range(self.pathnet.depth):
            path[i] = list(set(path[i]))
            path[i] = sorted(path[i])

        return path

    def sort_generation_by_fitness(self, population, fitness):
        fitness, population = zip(*list(reversed(sorted(list(zip(fitness, population))))))

        population = list(population)
        fitness = list(fitness)

        return population, fitness

    def simple_selection(self, population, fitness):
        return population[:int(len(population)/2)], fitness[:int(len(population)/2)]

    def simple_crossover(self, population):
        new_pop = []
        while len(new_pop) != len(population):
            father = random.choice(population)
            mother = random.choice(population)

            child = []
            for i in range(len(father)):
                layer = []
                if i % 2 == 0:
                    for modules in mother[i]:
                        layer.append(modules)
                else:
                    for modules in father[i]:
                        layer.append(modules)
                child.append(layer)
            new_pop.append(child)

        assert len(new_pop) == len(population), 'Simple_crossover(EA): new population not correct size'
        return new_pop
