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
        peptides: List[str], # we assume that each peptide is of same length l
        motifs: List[str], # we also assume that the motifs are of same length l
        t: int, # threshold
        seed: int | None = None,
    ):
        self.peptides = peptides
        self.motifs = motifs
        self.t = t
        self.l = len(self.peptides[0]) # peptide length
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def run(self):
        remaining_motifs = set(self.motifs)
        reactions_map = self._self_reactive_pairs()
        selected_peptides = []

        while remaining_motifs:
            
            # calculate for each peptide the number of reactions with the remaining self-reactive TCRs
            peptide_deletion_counts = {}
            for peptide in self.peptides:
                # calculate all reactions the peptide has on the remaining motifs set
                deletable = reactions_map[peptide] & remaining_motifs # TODO improve speed?
                peptide_deletion_counts[peptide] = len(deletable)
            
            # find the self peptide that deletes the most of these remaining self-reactive TCRs
            max_deleted = max(peptide_deletion_counts.values())
            if max_deleted == 0:
                break

            top_peptides = []
            for peptide, count in peptide_deletion_counts.items():
                if count == max_deleted:
                    top_peptides.append(peptide)
            
            # if multiple self peptides delete an equal number of remaining TCRs, we pick only those self peptides that 
            # do not overlap in the TCRs they delete
            deleted_this_round = set()
            chosen_peptides = []
            for peptide in top_peptides:
                deletable = reactions_map[peptide] & remaining_motifs
                if deleted_this_round.isdisjoint(deletable):
                    chosen_peptides.append(peptide)
                    deleted_this_round.update(deletable)

            # actually add the peptides to the list
            selected_peptides.extend(chosen_peptides)
            
            # remove all deleted motifs from the set of remaining motifs
            for peptide in chosen_peptides:
                remaining_motifs -= reactions_map[peptide]

        return selected_peptides
        
    def _affinity(self, motif: str, peptide: str) -> int:
        max_adjacent = 0
        current = 0
        for m, p in zip(motif, peptide):
            if m == p:
                current += 1
                if current > max_adjacent:
                    max_adjacent = current
            else:
                current = 0
        return max_adjacent
    
    def _self_reactive_pairs(self):
        motif_peptide_map = {}
        for peptide in self.peptides:
            motif_peptide_map[peptide] = set()
            for motif in self.motifs:
                if self._affinity(motif, peptide) >= self.t:
                    motif_peptide_map[peptide].add(motif)
        return motif_peptide_map
