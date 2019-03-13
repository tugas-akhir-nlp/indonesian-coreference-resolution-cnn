from abc import ABC, abstractmethod
from typing import List, Tuple

from utils.data_structures import UFDS
from utils.search import strict_binary_search


class TrainingInstancesGenerator(ABC):
    @abstractmethod
    def generate(self, training_ids: List[int], ufds: UFDS) -> List[Tuple[int, int, int]]:
        pass


class BudiInstancesGenerator(TrainingInstancesGenerator):
    def generate(self, training_ids: List[int], ufds: UFDS) -> List[Tuple[int, int, int]]:
        instances = []

        for a in range(len(training_ids)):
            for b in range(a + 1, len(training_ids)):
                instances.append(
                    (training_ids[a], training_ids[b], int(ufds.is_same(training_ids[a], training_ids[b])))
                )

        return instances


class SoonInstancesGenerator(TrainingInstancesGenerator):
    def generate(self, training_ids: List[int], ufds: UFDS) -> List[Tuple[int, int, int]]:
        instances = []

        chains = ufds.get_chain_list()

        for chain in chains:
            for i in range(len(chain) - 1):
                instances.append((chain[i], chain[i + 1], 1))

                antecedent_idx = strict_binary_search(training_ids, chain[i])
                anaphora_idx = strict_binary_search(training_ids, chain[i + 1])

                for j in range(antecedent_idx + 1, anaphora_idx):
                    instances.append((training_ids[j], chain[i + 1], 0))

        return instances


class GilangInstancesGenerator(TrainingInstancesGenerator):
    def generate(self, training_ids: List[int], ufds: UFDS) -> List[Tuple[int, int, int]]:
        instances = []
        added_pair = set()

        chains = ufds.get_chain_list()

        for chain in chains:
            for i in range(len(chain) - 1):
                instances.append((chain[i], chain[i + 1], 1))

                antecedent_idx = strict_binary_search(training_ids, chain[i])
                anaphora_idx = strict_binary_search(training_ids, chain[i + 1])

                for j in range(antecedent_idx, anaphora_idx + 1):
                    for k in range(j + 1, anaphora_idx + 1):
                        if not ufds.is_same(j, k) and not (j, k) in added_pair:
                            instances.append((training_ids[j], training_ids[k], 0))
                            added_pair.add((j, k))

        return instances
