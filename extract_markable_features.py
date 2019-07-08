import csv
import logging
from typing import Dict, List
from xml.etree import ElementTree

from utils.data_helper import (get_document_id_variables,
                               get_phrases_and_nodes, get_sentence_variables)
from utils.data_structures import UFDS
from utils.feature_extractors import SingleSyntacticFeatureExtractor


def is_singleton(ufds: UFDS, chain_dict: Dict[int, List[int]], node: int) -> int:
    par = ufds.root(node)
    return int(len(chain_dict[par]) == 1)


def is_antecedentless(ufds: UFDS, chain_dict: Dict[int, List[int]], node: int) -> int:
    par = ufds.root(node)
    return int(min(chain_dict[par]) == node)


def save_markable_features(markables: List[dict], output_file: str) -> None:
    if len(markables) == 0:
        return

    with open(output_file, 'w') as f:
        csv_file = csv.DictWriter(f, fieldnames=markables[0].keys())
        csv_file.writeheader()
        csv_file.writerows(markables)


def extract_markable_features(input_file: str, document_id_by_sentence_id: Dict[int, int]) -> List[Dict]:
    data = ElementTree.parse(input_file)
    root = data.getroot()
    parent_map = {c: p for p in root.iter() for c in p}

    logging.info('Getting phrases and nodes list')
    ufds = UFDS()
    phrases, nodes, phrase_id_by_node_id = get_phrases_and_nodes(ufds, root)
    chain_dict = ufds.get_chain_dict()

    feature_extractor = SingleSyntacticFeatureExtractor(
        phrases=phrases,
        document_id_by_sentence_id=document_id_by_sentence_id,
        parent_map=parent_map,
        phrase_id_by_node_id=phrase_id_by_node_id,
        context_words_count=10
    )

    logging.info('Extracting markable features')

    markables = [
        {
            'id': node.attrib['id'],
            'text': node.text,
            **feature_extractor.get_features(node),
            'is_singleton': is_singleton(ufds, chain_dict, int(node.attrib['id'])),
            'is_antecedentless': is_antecedentless(ufds, chain_dict, int(node.attrib['id']))
        }
        for node in nodes.values()
    ]

    logging.info('Finished extracting markable features')

    return markables


def main() -> None:
    _, markable_ids_by_sentence_id = get_sentence_variables('data/full.xml')
    document_id_by_sentence_id, _, _, _ = get_document_id_variables(
        'data/document_id.csv', markable_ids_by_sentence_id)

    logging.info('Extracting training markable features...')
    training_markables = extract_markable_features(
        'data/training/data.xml', document_id_by_sentence_id)

    logging.info('Extracting testing markable features...')
    testing_markables = extract_markable_features(
        'data/testing/data.xml', document_id_by_sentence_id)

    logging.info('Saving training markable features...')
    save_markable_features(training_markables, 'data/training/markables.csv')

    logging.info('Saving testing markable features...')
    save_markable_features(testing_markables, 'data/testing/markables.csv')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    main()
