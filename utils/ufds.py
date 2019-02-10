from typing import Hashable, Set, Dict, List


class UFDS:
    def __init__(self):
        self.nodes: Set[Hashable] = set()
        self.parent: Dict[Hashable, Hashable] = {}

    def init_id(self, *nodes: Hashable) -> None:
        for node in nodes:
            if node not in self.nodes:
                self.nodes.add(node)
                self.parent[node] = node

    def root(self, x: Hashable) -> Hashable:
        self.init_id(x)

        if self.parent[x] == x:
            return x
        else:
            self.parent[x] = self.root(self.parent[x])
            return self.parent[x]

    def gabung(self, x: Hashable, y: Hashable) -> None:
        self.init_id(x, y)

        self.parent[self.root(x)] = self.root(y)

    def is_same(self, x: Hashable, y: Hashable) -> bool:
        self.init_id(x, y)

        return self.root(x) == self.root(y)

    def get_chains(self) -> List[List[Hashable]]:
        leluhur = [x for x in self.nodes if self.root(x) == x]
        chains = [
            [y for y in self.nodes if self.is_same(x, y)] for x in leluhur]

        return chains
