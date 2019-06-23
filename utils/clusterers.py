from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Tuple, Dict, Iterable, Set

import numpy as np

from utils.data_structures import UFDS


def get_anaphora_scores_by_antecedent(m1_ids: Iterable[int], m2_ids: Iterable[int], scores: np.ndarray,
                                      singletons: Set[int] = None) -> Dict[int, List[Tuple[int, float]]]:
    if singletons is None:
        singletons = set()

    anaphora_scores_by_antecedent = {}

    for m1_id, m2_id, score in zip(m1_ids, m2_ids, scores):
        if m1_id not in anaphora_scores_by_antecedent and m1_id not in singletons:
            anaphora_scores_by_antecedent[m1_id] = []

        if m2_id not in anaphora_scores_by_antecedent and m2_id not in singletons:
            anaphora_scores_by_antecedent[m2_id] = []

        if m1_id not in singletons and m2_id not in singletons:
            anaphora_scores_by_antecedent[m1_id].append((m2_id, score))

    return anaphora_scores_by_antecedent


class Clusterer(ABC):
    def get_chains(self, anaphora_scores_by_antecedent: Dict[int, List[Tuple[int, float]]], threshold: float = 0.5) -> \
            List[List[int]]:
        ufds = UFDS()

        for m1_id, m2_ids in anaphora_scores_by_antecedent.items():
            ufds.init_id(m1_id)
            passed_anaphora = list(filter(lambda x: x[1][1] > threshold, m2_ids))
            choosed_pairs = self._choose_pairs(m1_id, passed_anaphora)

            for choosed_pair in choosed_pairs:
                ufds.join(m1_id, choosed_pair)

        return ufds.get_chain_list()

    @abstractmethod
    def _choose_pairs(self, m1_id: int, anaphora_scores: List[Tuple[int, float]]) -> List[int]:
        pass


class BestFirstClusterer(Clusterer):
    def _choose_pairs(self, m1_id: int, anaphora_scores: List[Tuple[int, float]]) -> List[int]:
        if len(anaphora_scores) == 0:
            return []

        best_anaphora = reduce(lambda x, y: x if x[1][1] > y[1][1] else y, anaphora_scores, (-1, [-1, -1]))
        return [best_anaphora[0]]


class ClosestFirstClusterer(Clusterer):
    def _choose_pairs(self, m1_id: int, anaphora_scores: List[Tuple[int, float]]) -> List[int]:
        if len(anaphora_scores) == 0:
            return []

        anaphora_distance = map(lambda x: (x[0], abs(x[0] - m1_id)), anaphora_scores)
        closest_anaphora = reduce(lambda x, y: x if x[1] < y[1] else y, anaphora_distance, (-1, 999999))
        return [closest_anaphora[0]]
