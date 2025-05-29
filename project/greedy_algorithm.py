from __future__ import annotations

import numpy as np
import random
from typing import List
import math

class GreedyAlgorithm:
    '''A greedy optimizer implementation based on the paper 'Is T Cell Negative Selection a Learning Algorithm?' 
    (Wortel et al., 2020).
    '''
    def __init__(
        self,
        peptides: List[str], # we assume that each peptide is of same length
        t: int, # threshold
        seed: int | None = None,
    ):
        self.peptides = peptides
        self.t = t
        self.l = len(self.peptides[0]) # peptide length
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def run(self):
        pass
        
    def _affinity(self, p1: str, p2: str):
        max_adjacent = 0
        current = 0
        for a, b in zip(p1, p2):
            if a == b:
                current += 1
                if current > max_adjacent:
                    max_adjacent = current
            else:
                current = 0
        return max_adjacent
    