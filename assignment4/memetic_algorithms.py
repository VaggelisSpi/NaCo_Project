import pandas as pd
import numpy as np
from typing import List
import random


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
    p: crossover rate
    m: mutation rate
    fitness_function: function determining the fitness
    K: tournament selection parameter
    generations: amount of iterations
    '''
    # start with random population of candidate solutions
    population = np.array([np.random.permutation(range(sigma)) for _ in range(N)])

    for _ in range(generations):
        # determine fitness of every solution
        fitness = fitness_pop(population, cities)

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

    return population


def fitness_pop(population: List[List[float]], cities: List[float]) -> List[float]:
    distances = np.zeros(len(population))

    for i, candidate in enumerate(population):
        distances[i] = fitness(candidate, cities)

    return 1 / distances


def fitness(candidate: List[float], cities: List[float]) -> float:
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


def memetic():
    pass


def two_opt(candidate: List[float], cities: List[float]) -> List[float]:
    best = candidate.copy()
    search = True

    while search:
        search = False
        for i in range(1, len(candidate) - 2):  # skip the first and last city
            for j in range(i+1, len(candidate)):
                if (j - i) > 1:
                    new_candidate = best.copy()
                    new_candidate[i:j] = new_candidate[i:j][::-1]
                    if fitness(new_candidate, cities) > fitness(best, cities):
                        best = new_candidate
                        search = True

    return best


if __name__ == "__main__":
    cities = np.loadtxt("./assignment4/file-tsp.txt")
    population = simple_EA(cities, 50, 100, 20)
    print(population)

    cities = np.loadtxt("./assignment4/berlin52.txt")[:, 1:]
    population = simple_EA(cities, 52, 100, 20)
    print(population)
