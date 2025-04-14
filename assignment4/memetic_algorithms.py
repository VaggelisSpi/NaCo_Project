import numpy as np
from typing import List
import random
import matplotlib.pyplot as plt
import time


def simple_EA(
        cities: List[float],
        sigma: int,
        N: int,
        K: int,
        p: float = 0.1,
        m: float = 0.01,
        generations: int = 1000):
    '''
    cities: list with city coordinates
    sigma: alphabet (size)
    N: population size
    K: tournament selection parameter
    p: crossover rate
    m: mutation rate
    generations: amount of iterations
    '''
    fitness_avg = []
    fitness_best = []
    # start with random population of candidate solutions
    population = np.array([np.random.permutation(range(sigma)) for _ in range(N)])

    for _ in range(generations):
        # determine fitness of every solution
        fitness = fitness_pop(population, cities)
        fitness_avg.append(fitness.sum() / len(fitness))
        fitness_best.append(max(fitness))

        # select the new population
        new_population = []
        for i in range(N):
            parent1, parent2 = population[tournament(N, K, fitness)]

            # crossover
            if random.random() < p:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # mutation
            if random.random() < m:
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]

            new_population.append(child)

        population = np.array(new_population)

    fitness = fitness_pop(population, cities)
    fitness_avg.append(fitness.sum() / len(fitness))
    fitness_best.append(max(fitness))

    return fitness_avg, fitness_best


def fitness_pop(population: List[List[float]], cities: List[float]) -> List[float]:
    distances = np.zeros(len(population))

    for i, candidate in enumerate(population):
        distances[i] = fitness(candidate, cities)

    return 1 / distances


def fitness(candidate: List[int], cities: List[float]) -> float:
    distance = 0

    for i in range(len(candidate)-1):
        x1, y1 = cities[candidate[i]]
        x2, y2 = cities[candidate[i+1]]
        distance += np.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)

    return distance


def tournament(N: int, K: int, fitness: List[str]) -> List[int]:
    parent1_idx = np.random.choice(N, K, replace=False)
    parent1_idx = np.argmax(fitness[parent1_idx])

    parent2_idx = np.random.choice(N, K, replace=False)
    parent2_idx = np.argmax(fitness[parent2_idx])

    return [parent1_idx, parent2_idx]


def crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    size = len(parent1)
    child = [None] * size

    # determine cut points
    start, end = sorted(random.sample(range(size), 2))

    # keep middle piece
    child[start:end+1] = parent1[start:end+1]

    # take complement in other parent
    to_do = parent2.copy()
    for city in child:
        to_do = to_do[to_do != city]

    # fill gaps in order
    child[end+1:] = to_do[:size-(end+1)]
    child[:start] = to_do[size-(end+1):]

    return child


def memetic(
        cities: List[float],
        sigma: int,
        N: int,
        K: int,
        p: float = 0.1,
        m: float = 0.1,
        generations: int = 1000):
    '''
    cities: list with city coordinates
    sigma: alphabet (size)
    N: population size
    K: tournament selection parameter
    p: crossover rate
    m: mutation rate
    generations: amount of iterations
    '''
    fitness_avg = []
    fitness_best = []

    # start with random population of candidate solutions
    population = np.array([np.random.permutation(range(sigma)) for _ in range(N)])

    # apply local search
    for i, candidate in enumerate(population):
        population[i] = two_opt(candidate, cities)

    for _ in range(generations):
        # select the new population
        new_population = []

        # determine fitness
        fitness = fitness_pop(population, cities)
        fitness_avg.append(fitness.sum() / len(fitness))
        fitness_best.append(max(fitness))

        for i in range(N):
            parent1, parent2 = population[tournament(N, K, fitness)]

            # crossover
            if random.random() < p:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # mutation
            if random.random() < m:
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]

            child = two_opt(child, cities)
            new_population.append(child)

        population = np.array(new_population)

    fitness = fitness_pop(population, cities)
    fitness_avg.append(fitness.sum() / len(fitness))
    fitness_best.append(max(fitness))

    return fitness_avg, fitness_best


def two_opt(candidate: List[int], cities: List[float]) -> List[int]:
    best = candidate.copy()
    search = True

    while search:
        search = False
        for i in range(1, len(candidate) - 2):  # skip the first and last city
            for j in range(i+1, len(candidate)):
                if (j - i) > 1:
                    new_candidate = best.copy()
                    new_candidate[i:j] = new_candidate[i:j][::-1]
                    if fitness(new_candidate, cities) < fitness(best, cities):
                        best = new_candidate
                        search = True

    return best


def make_plot(algorithm: str, sigma: int, N: int, K: int, G: int, fitness_avg: List[float], fitness_best: List[float]):
    name = algorithm + '_' + str(sigma) + '_' + str(N) + '_' + str(K) + '_' + str(G) + '.png'

    plt.figure()
    plt.plot(range(len(fitness_avg)), 1/np.array(fitness_avg), label='average distance')
    plt.plot(range(len(fitness_best)), 1/np.array(fitness_best), label='minimal distance')
    plt.legend()
    plt.xlabel('Generations')
    plt.ylabel('Distance (Euclidean)')
    plt.savefig(name)


if __name__ == "__main__":
    cities = np.loadtxt("./assignment4/file-tsp.txt")
    sigma = len(cities)

    N = 300
    K = 100
    G = 500
    start_time = time.time()
    fitness_avg, fitness_best = simple_EA(cities, sigma, N, K, generations=G)
    end_time = time.time()
    print(f'time: {end_time-start_time}')
    print(f'last distance: {1/fitness_best[-1]}\n')
    make_plot('EA', sigma, N, K, G, fitness_avg, fitness_best)

    N = 10
    K = 4
    G = 1
    start_time = time.time()
    fitness_avg, fitness_best = memetic(cities, sigma, N, K)
    end_time = time.time()
    print(f'time: {end_time-start_time}')
    print(f'last distance: {1/fitness_best[-1]}\n')
    make_plot('MA', sigma, N, K, G, fitness_avg, fitness_best, generations=G)

    cities = np.loadtxt("./assignment4/burma14.txt")[:, 1:]
    sigma = len(cities)

    N = 100
    K = 20
    G = 500
    start_time = time.time()
    fitness_avg, fitness_best = simple_EA(cities, sigma, N, K, generations=G)
    end_time = time.time()
    print(f'time: {end_time-start_time}')
    print(f'last distance: {1/fitness_best[-1]}\n')
    make_plot('EA', sigma, N, K, G, fitness_avg, fitness_best)

    N = 50
    K = 10
    G = 5
    start_time = time.time()
    fitness_avg, fitness_best = memetic(cities, sigma, N, K, generations=G)
    end_time = time.time()
    print(f'time: {end_time-start_time}')
    print(f'last distance: {1/fitness_best[-1]}\n')
    make_plot('MA', sigma, N, K, G, fitness_avg, fitness_best)
