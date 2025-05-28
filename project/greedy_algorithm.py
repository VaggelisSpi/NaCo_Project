import numpy as np
import random
from typing import List
import math
from collections.abc import Callable


def greedy_alg(data: List[float], sigma: int, fitness: Callable):
    data = np.array(data)
    final = []
    old = list(range(len(data)))

    while len(final) < sigma:
        best_index = None
        best_score = float('inf')

        for index in old:
            new = final + [index]
            score = fitness(data[new])
            if score < best_score:
                best_score = score
                best_index = index

        final.append(best_index)
        old.remove(best_index)

    return final
