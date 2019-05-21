from collections import Counter
from typing import List, Dict, Tuple

from sklearn.utils.linear_assignment_ import linear_assignment


class PerDocumentScorer:
    def __init__(self, document_id_by_markable_id: Dict[int, int], metric: str) -> None:
        self.document_id_by_markable_id = document_id_by_markable_id

        if metric == 'muc':
            self.metric = muc
        elif metric == 'b3':
            self.metric = b_cubed
        elif metric == 'ceafe':
            self.metric = ceafe

    def get_scores(self, predicted_chains: List[List[int]], label_chains: List[List[int]]) -> \
            Tuple[float, float, float]:
        evaluator = Evaluator(self.metric)

        predicted_chains_by_document_id = self._get_chains_by_document_id(predicted_chains)
        label_chains_by_document_id = self._get_chains_by_document_id(label_chains)
        markable_id_to_predicted_chain = self._get_markable_id_to_chain(predicted_chains)
        markable_id_to_label_chain = self._get_markable_id_to_chain(label_chains)

        for document_id in label_chains_by_document_id.keys():
            document = Document(predicted_chains_by_document_id[document_id], label_chains_by_document_id[document_id],
                                markable_id_to_predicted_chain, markable_id_to_label_chain)

            evaluator.update(document)

        return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()

    def _get_chains_by_document_id(self, chains: List[List[int]]) -> Dict[int, List[List[int]]]:
        chains_by_document_id = {}

        for chain in chains:
            document_id = self.document_id_by_markable_id[chain[0]]

            if document_id not in chains_by_document_id.keys():
                chains_by_document_id[document_id] = []

            chains_by_document_id[document_id].append(chain)

        return chains_by_document_id

    def _get_markable_id_to_chain(self, chains: List[List[int]]) -> Dict[int, List[int]]:
        markable_id_to_chain = {}

        for chain in chains:
            for markable_id in chain:
                markable_id_to_chain[markable_id] = chain

        return markable_id_to_chain


class Document:
    def __init__(self, clusters, gold, mention_to_cluster, mention_to_gold):
        self.clusters = clusters
        self.gold = gold
        self.mention_to_cluster = mention_to_cluster
        self.mention_to_gold = mention_to_gold


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class Evaluator:
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, document):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(document.clusters, document.gold)
        else:
            pn, pd = self.metric(document.clusters, document.mention_to_gold)
            rn, rd = self.metric(document.gold, document.mention_to_cluster)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.iteritems():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem
