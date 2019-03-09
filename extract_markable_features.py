import csv
import logging
from typing import Dict, List
from xml.etree import ElementTree

from utils.data_helper import get_phrases_and_nodes
from utils.data_structures import UFDS
from utils.feature_extractors import SingleSyntacticFeatureExtractor


def is_singleton(ufds: UFDS, chain_dict: Dict[int, List[int]], node: int) -> int:
    par = ufds.root(node)
    return int(len(chain_dict[par]) == 1)


def save_markable_features(markables: List[dict], output_file: str) -> None:
    if len(markables) == 0:
        return

    with open(output_file, 'w') as f:
        csv_file = csv.DictWriter(f, fieldnames=markables[0].keys())
        csv_file.writeheader()
        csv_file.writerows(markables)


def extract_markable_features(input_file: str) -> List[dict]:
    data = ElementTree.parse(input_file)
    root = data.getroot()

    logging.info('Getting phrases and nodes list')
    ufds = UFDS()
    phrases, nodes, phrase_id_by_node_id = get_phrases_and_nodes(ufds, root)
    chain_dict = ufds.get_chain_dict()

    feature_extractor = SingleSyntacticFeatureExtractor(
        phrases=phrases,
        phrase_id_by_node_id=phrase_id_by_node_id,
        context_words_count=10
    )

    logging.info('Extracting markable features')

    markables = [
        {
            'id': node.attrib['id'],
            'text': node.text,
            **feature_extractor.get_features(node),
            'is_singleton': is_singleton(ufds, chain_dict, int(node.attrib['id']))
        }
        for node in nodes.values()
    ]

    logging.info('Finished extracting markable features')

    return markables


def main() -> None:
    logging.info('Extracting training markable features...')
    training_markables = extract_markable_features('data/training/data.xml')

    logging.info('Extracting testing markable features...')
    testing_markables = extract_markable_features('data/testing/data.xml')

    logging.info('Saving training markable features...')
    save_markable_features(training_markables, 'data/training/markables.csv')

    logging.info('Saving testing markable features...')
    save_markable_features(testing_markables, 'data/testing/markables.csv')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    main()
