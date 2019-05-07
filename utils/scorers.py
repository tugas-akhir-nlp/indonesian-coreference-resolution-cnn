from abc import ABC, abstractmethod
from collections import Counter
from functools import reduce
from typing import List, Tuple, Dict


class Scorer(ABC):
    precision: float
    recall: float

    def get_scores(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) \
            -> Tuple[float, float, float]:
        self._clear_memo()

        precision = self._compute_precision(predicted_chains, label_chains)
        recall = self._compute_recall(predicted_chains, label_chains)
        f1 = self._compute_f1(predicted_chains, label_chains)

        return precision, recall, f1

    def _clear_memo(self):
        self._precision = None
        self._recall = None

    def _compute_f1(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        precision = self._compute_precision(predicted_chains, label_chains)
        recall = self._compute_recall(predicted_chains, label_chains)

        if precision + recall == 0:
            return 0

        return 2 * precision * recall / (precision + recall)

    def _compute_precision(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        if self._precision is None:
            self._precision = self.compute_precision(predicted_chains, label_chains)

        return self._precision

    def _compute_recall(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        if self._recall is None:
            self._recall = self.compute_recall(predicted_chains, label_chains)

        return self._recall

    @abstractmethod
    def compute_precision(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        pass

    @abstractmethod
    def compute_recall(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        pass


class MUCScorer(Scorer):
    def compute_precision(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        return self._general_compute(predicted_chains, label_chains)

    def compute_recall(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        return self._general_compute(label_chains, predicted_chains)

    def _general_compute(self, chain1: List[List[int]], chain2: List[List[int]]) -> float:
        nominator = 0
        denominator = 0

        for c1 in chain1:
            ki = len(c1)
            part_left = ki
            partition = 0

            for c2 in chain2:
                found = False

                for s in c2:
                    if s in c1:
                        found = True
                        part_left -= 1

                if found:
                    partition += 1

            nominator += (ki - (partition + part_left))
            denominator += ki - 1

        if denominator == 0:
            return 0

        return nominator / denominator


class B3Scorer(Scorer):
    def compute_precision(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        return self._general_compute(predicted_chains, label_chains)

    def compute_recall(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        return self._general_compute(label_chains, predicted_chains)

    def _general_compute(self, chain1: List[List[int]], chain2: List[List[int]]) -> float:
        mention_to_gold = {}

        for c in chain2:
            for m in c:
                mention_to_gold[m] = c

        num, dem = 0, 0

        for c in chain1:
            if len(c) == 1:
                continue

            gold_counts = Counter()
            correct = 0
            for m in c:
                if m in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[m])] += 1
            for c2, count in gold_counts.items():
                if len(c2) != 1:
                    correct += count * count

            num += correct / float(len(c))
            dem += len(c)

        if dem == 0:
            return 0

        return num / dem


class CEAFeScorer(Scorer):
    def compute_precision(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        return self._general_compute(predicted_chains, label_chains)

    def compute_recall(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        return self._general_compute(label_chains, predicted_chains)
    
    def _general_compute(self, chain1: List[List[int]], chain2: List[List[int]]) -> float:
        key_to_value: Dict[int, int] = {}
        value_to_key: Dict[int, int] = {}
        key_to_similar_mention: Dict[int, int] = {}

        for idx_c1 in range(len(chain1)):
            c1 = chain1[idx_c1]

            max_similarity = 0
            idx_similar = None

            for idx_c2 in range(len(chain2)):
                c2 = chain2[idx_c2]

                similar_mention = 0

                for m2 in c2:
                    if m2 in c1:
                        similar_mention += 1

                if similar_mention > max_similarity:
                    max_similarity = similar_mention
                    idx_similar = idx_c2

            if idx_similar is not None:
                if idx_similar in key_to_value.values():
                    if max_similarity > key_to_similar_mention[value_to_key[idx_similar]]:
                        key_to_value.pop(value_to_key[idx_similar])
                        key_to_similar_mention.pop(value_to_key[idx_similar])

                        key_to_value[idx_c1] = idx_similar
                        value_to_key[idx_similar] = idx_c1
                        key_to_similar_mention[idx_c1] = max_similarity
                else:
                    key_to_value[idx_c1] = idx_similar
                    value_to_key[idx_similar] = idx_c1
                    key_to_similar_mention[idx_c1] = max_similarity

        sum_similar = 0
        for idx_key in key_to_value.keys():
            sum_similar += 2 * key_to_similar_mention[idx_key] / (
                    len(chain1[idx_key]) + len(chain2[key_to_value[idx_key]]))

        return sum_similar / len(chain1)


class AverageScorer(Scorer):
    score: float

    def __init__(self, scorers: List[Scorer]):
        self.scorers = scorers

    def get_scores(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) \
            -> Tuple[float, float, float]:
        self.score = None
        return super().get_scores(predicted_chains, label_chains)

    def compute_precision(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        return self._compute_score(predicted_chains, label_chains)

    def compute_recall(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        return self._compute_score(predicted_chains, label_chains)

    def _compute_score(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> float:
        if self.score is not None:
            return self.score

        if len(self.scorers) == 0:
            self.score = 0
            return self.score

        sum_f1 = reduce(lambda prv, scorer: prv + scorer.get_scores(predicted_chains, label_chains)[2], self.scorers, 0)
        self.score = sum_f1 / len(self.scorers)

        return self.score
