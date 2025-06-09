from __future__ import annotations

import numpy as np
import random
from typing import List
import math
from collections.abc import Callable


class EA:
    """This class can be used for dataset optimization. A population is defined as a list of indices that refer back to
    the original datapoints.
    """

    def __init__(
        self,
        data: List[float],
        N: int,  # population size
        sigma: int,  # individual size
        K: int,  # tournament selection
        f: Callable,  # fitness function
        p: float = 0.1,  # crossover rate
        mu: float = 0.01,  # mutation rate
        T: int = 1000,  # number of iterations
        seed: int | None = None,
    ):
        self.data = np.array(data)
        self.N = N
        self.sigma = sigma
        self.K = K
        self.p = p
        self.mu = mu
        self.T = T
        self.f = f

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if self.N % 2 != 0:  # make N even to prevent possible index out of bounds errors
            self.N += 1

        self.M = len(self.data)  # total data keys
        self.population: List[float] = []
        self.best_fitness = np.inf
        self.best_individual = None

    # TODO Implement a stop condition?
    def run(self):
        fitness_avg = []
        fitness_best = []
        self.population = [self._generate_individual() for _ in range(self.N)]

        for t in range(self.T):
            # get fitness values of the populations
            fitnesses = [self.f(individual) for individual in self.population]

            # statistics
            best_idx = np.argmin(fitnesses)
            best_fit = fitnesses[best_idx]
            avg_fit = float(np.mean(fitnesses))

            if best_fit < self.best_fitness:
                self.best_fitness = best_fit
                self.best_individual = self.population[best_idx]

            fitness_avg.append(avg_fit)
            fitness_best.append(best_fit)

            if t % 100 == 0:
                print(f"Gen {t + 1}: Best = {best_fit}, Avg = {avg_fit}")

            new_population = []
            i = 0
            while i < self.N - 1:  # so it loops self.N // 2 times
                parent1 = self._tournament(fitnesses)
                parent2 = self._tournament(fitnesses)

                # sometimes do crossover
                if random.random() < self.p:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1 = parent1[:]
                    child2 = parent2[:]

                # always mutate
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.extend([child1, child2])

                i += 2

            self.population = new_population

        return (self.best_individual, self.best_fitness, fitness_avg, fitness_best)

    def _generate_individual(self) -> List[float]:
        return random.sample(range(self.M), self.sigma)

    def _tournament(self, fitnesses: List[float]):
        selected = np.random.choice(self.N, self.K, replace=False)
        best = selected[np.argmin([fitnesses[i] for i in selected])]
        return self.population[best]

    # FIX? The order of a population is not really meaningful, so I am not 100% sure about this.
    def _crossover(self, p1: List[int], p2: List[int]):
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))

        child1 = [None] * size
        child2 = [None] * size

        child1[start : end + 1] = p1[start : end + 1]
        child2[start : end + 1] = p2[start : end + 1]

        def fill(child, donor, slice_vals):
            insert_positions = list(range(end + 1, size)) + list(range(0, start))
            values_to_insert = [val for val in donor if val not in slice_vals]
            for pos, val in zip(insert_positions, values_to_insert):
                child[pos] = val

        fill(child1, p2, p1[start : end + 1])
        fill(child2, p1, p2[start : end + 1])

        return child1, child2

    # FIX? We can't use the same mutate logic as shown in lecture 6 about TCP because we have a different problem.
    # Therefore, I am am not sure if this is what we want, as it's not exactly motivated by literature.
    def _mutate(self, p: List[int]):
        mutated = p[:]
        num_mutations = math.ceil(self.mu * self.sigma)

        if num_mutations <= 0:
            return mutated

        # select indices to replace
        to_replace = random.sample(mutated, num_mutations)

        # get indices that are not in the individual
        available = list(set(range(self.M)) - set(mutated))
        replacements = random.sample(available, num_mutations)

        # replace selection
        for old, new in zip(to_replace, replacements):
            i = mutated.index(old)
            mutated[i] = new

        return mutated
