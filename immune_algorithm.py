import math
import random
import time
import numpy as np


class ImmuneAlgorithm(object):

    def __init__(self, file=None, pop_size=100, gen=100, p_c=0.1, p_m=0.5, n_best=20):
        self.pop_size = pop_size
        self.gen = gen
        self.p_c = p_c
        self.p_m = p_m
        self.n_best = n_best
        self.best_chromosome = np.array([])
        self.best_fit = math.inf
        if file is not None:
            with open(file) as fp:
                for i, line in enumerate(fp):
                    if i == 0:
                        self.n = int(line[2:])
                        self.cost_matrix = np.empty((0, self.n), int)
                        self.distance_matrix = np.empty((0, self.n), int)
                    elif 1 < i < self.n + 2:
                        # wczytanie macierzy przepływu
                        self.distance_matrix = np.append(self.distance_matrix, [list(map(int, line[2:].split('  ')))],
                                                         axis=0)
                    elif self.n + 2 < i < (2 * self.n) + 3:
                        # wczytanie macierzy odległości
                        self.cost_matrix = np.append(self.cost_matrix, [list(map(int, line[2:].split('  ')))], axis=0)
                    if i == (2 * self.n) + 3:
                        break
        if file is None:
            self.n = 5
            self.distance_matrix = np.array(
                [[0, 4, 5, 1, 4], [1, 0, 5, 6, 7], [4, 6, 0, 1, 5], [3, 6, 7, 0, 2], [4, 6, 2, 5, 0]])
            self.cost_matrix = np.array(
                [[0, 4, 5, 1, 4], [1, 0, 5, 6, 7], [4, 6, 0, 1, 5], [3, 6, 7, 0, 2], [4, 6, 2, 5, 0]])
        self.cur_pop = None
        self.evaluated_pop = None
        self.selected_pop = None
        self.min_fitness = None
        self.max_fitness = math.inf
        self.fraction_difference_sum = 0
        self.new_pop = None
        self.worst_history = np.array([])
        self.avg_history = np.array([])
        self.best_history = np.array([])
        pass

    def initialize(self):
        start_pop = np.array([np.arange(1, self.n + 1) for i in range(self.pop_size)])
        for i in start_pop:
            np.random.shuffle(i)
        self.cur_pop = start_pop

    def evaluate(self, chromosome):
        return sum(np.sum(self.cost_matrix * self.distance_matrix[chromosome[:, None] - 1, chromosome - 1], 1))

    def evaluation(self):
        self.evaluated_pop = np.array([self.evaluate(i) for i in self.cur_pop])
        self.max_fitness = np.amin(self.evaluated_pop) if np.amin(
            self.evaluated_pop) < self.max_fitness else self.max_fitness
        self.min_fitness = np.amax(self.evaluated_pop)
        self.fraction_difference_sum = np.sum([1 / i for i in self.evaluated_pop])
        self.worst_history = np.append(self.worst_history, np.amax(self.evaluated_pop))
        self.avg_history = np.append(self.avg_history, np.average(self.evaluated_pop))
        self.best_history = np.append(self.best_history, np.amin(self.evaluated_pop))

    def proportional_prob(self, cur_fitness):
        return (1 / cur_fitness) / np.sum(self.fraction_difference_sum)

    def selection(self):
        self.selected_pop = np.empty((0, self.n), int)
        for selected_index in np.reshape(self.evaluated_pop.argsort()[:self.n_best],
                                         [self.evaluated_pop.argsort()[:self.n_best].shape[0], 1]):
            self.selected_pop = np.append(self.selected_pop,
                                          np.reshape(self.cur_pop[selected_index], [1, self.n]), axis=0)
        self.fraction_difference_sum = np.sum(np.divide(1, self.evaluated_pop))

    def cloning(self):
        self.new_pop = np.copy(self.selected_pop)
        for i in self.selected_pop:
            if np.random.random() <= self.p_c:
                self.new_pop = np.append(self.new_pop, np.reshape(i, [1, self.n]), axis=0)
            if len(self.new_pop) == self.pop_size:
                break
        # self.new_pop = np.reshape(self.new_pop, [int(len(self.new_pop) / self.n), self.n])

        selected_pop_probabilities = np.divide(1, np.array([self.evaluate(i) for i in self.selected_pop]))

        # uzupełnienie brakujących osobników w populacji
        while len(self.new_pop) < self.pop_size:
            for i in self.selected_pop:
                if np.random.uniform(min(selected_pop_probabilities),
                                     max(selected_pop_probabilities)) <= self.proportional_prob(
                        self.evaluate(i)):
                    self.new_pop = np.append(self.new_pop, np.reshape(i, [1, self.n]), axis=0)

        # uznanie nowej generacji jako obecnej
        self.new_pop = self.new_pop.astype(int)
        self.cur_pop = np.copy(self.new_pop)
        self.new_pop = None

    def mutation(self):
        for chromosome in self.cur_pop:
            for i, gene in enumerate(chromosome):
                if np.random.rand() <= self.p_m:
                    changing_gene = random.choice([j for j in range(0, self.n) if j != i])
                    chromosome[i], chromosome[changing_gene] = chromosome[changing_gene], chromosome[i]

    def run(self):
        best_fitnesses = np.array([])
        times = np.array([])
        for j in range(10):
            start_time = time.time()
            self.initialize()
            self.evaluation()
            for i in range(self.gen):
                self.selection()
                self.cloning()
                self.mutation()
                self.evaluation()
            print("Najlepsze przystosowanie: " + str(self.max_fitness))
            times = np.append(times, time.time() - start_time)
            best_fitnesses = np.append(best_fitnesses, self.max_fitness)
            self.max_fitness = math.inf
        print("Średnie najlepsze przystosowanie: " + str(np.average(best_fitnesses)))
        print("Średni czas obliczeń: " + str(np.average(times)))
        return np.average(best_fitnesses)

    def run_line_chart(self):
        start_time = time.time()
        self.initialize()
        self.evaluation()
        for i in range(self.gen):
            self.selection()
            self.cloning()
            self.mutation()
            self.evaluation()
        elapsed_time = time.time() - start_time
        print("Czas obliczeń: " + str(elapsed_time))
        generations = np.arange(self.gen + 1)
        return self.worst_history, self.avg_history, self.best_history, generations
