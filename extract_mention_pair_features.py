import csv
import logging
from typing import List, Type
from xml.etree import ElementTree

from utils.data_helper import get_phrases_and_nodes, get_sentence_variables, get_document_id_variables
from utils.data_structures import UFDS
from utils.feature_extractors import PairSyntacticFeatureExtractor, BudiFeatureExtractor
from utils.training_instances_generator import TrainingInstancesGenerator, BudiInstancesGenerator, \
    SoonInstancesGenerator, GilangInstancesGenerator


def save_mention_pair_features(mention_pairs: List[dict], output_file: str) -> None:
    if len(mention_pairs) == 0:
        return

    with open(output_file, 'w') as f:
        csv_file = csv.DictWriter(f, fieldnames=mention_pairs[0].keys())
        csv_file.writeheader()
        csv_file.writerows(mention_pairs)


def extract_mention_pair_features(input_file: str,
                                  instances_generator: TrainingInstancesGenerator,
                                  feature_extractor_class: Type[
                                      PairSyntacticFeatureExtractor] = PairSyntacticFeatureExtractor) -> List[dict]:
    data = ElementTree.parse(input_file)
    root = data.getroot()
    parent_map = {c: p for p in root.iter() for c in p}

    logging.info('Getting phrases and nodes list')
    ufds = UFDS()
    phrases, nodes, phrase_id_by_node_id = get_phrases_and_nodes(ufds, root)

    feature_extractor = feature_extractor_class(
        parent_map=parent_map,
        phrases=phrases,
        phrase_id_by_node_id=phrase_id_by_node_id
    )

    logging.info('Generating training instances')
    training_node_ids = sorted(list(ufds.nodes))
    training_instances = instances_generator.generate(training_node_ids, ufds)

    logging.info('Extracting mention pairs features')
    mention_pairs = [
        {
            'm1_id': instance[0],
            'm2_id': instance[1],
            **feature_extractor.get_features(nodes[instance[0]], nodes[instance[1]]),
            'is_coreference': instance[2],
        }
        for instance in training_instances
    ]

    return mention_pairs


def main() -> None:
    _, markable_ids_by_sentence_id = get_sentence_variables('data/full.xml')
    _, document_id_by_markable_id, _, _ = get_document_id_variables('data/document_id.csv', markable_ids_by_sentence_id)

    training_instances_generators = [
        ('soon', SoonInstancesGenerator()),
        ('gilang', GilangInstancesGenerator()),
        ('budi', BudiInstancesGenerator(document_id_by_markable_id))
    ]

    for name, generator in training_instances_generators:
        logging.info('Extracting features using %s\'s training instances generator...' % name)
        mention_pairs = extract_mention_pair_features('data/training/data.xml', generator)

        logging.info('Saving features using %s\'s training instances generator...' % name)
        save_mention_pair_features(mention_pairs, 'data/training/mention_pairs_%s.csv' % name)

    logging.info('Extracting training data for Budi et al. (2006) implementation...')
    mention_pairs = extract_mention_pair_features('data/training/data.xml',
                                                  BudiInstancesGenerator(document_id_by_markable_id),
                                                  BudiFeatureExtractor)

    logging.info('Saving training data for Budi et al. (2006) implementation...')
    save_mention_pair_features(mention_pairs, 'data/training/mention_pairs_for_budi_et_al_implementation.csv')

    logging.info('Extracting testing data features...')
    mention_pairs = extract_mention_pair_features('data/testing/data.xml',
                                                  BudiInstancesGenerator(document_id_by_markable_id))

    logging.info('Saving testing data features...')
    save_mention_pair_features(mention_pairs, 'data/testing/mention_pairs.csv')

    logging.info('Extracting testing data for Budi et al. (2006) implementation...')
    mention_pairs = extract_mention_pair_features('data/testing/data.xml',
                                                  BudiInstancesGenerator(document_id_by_markable_id),
                                                  BudiFeatureExtractor)

    logging.info('Saving testing data for Budi et al. (2006) implementation...')
    save_mention_pair_features(mention_pairs, 'data/testing/mention_pairs_for_budi_et_al_implementation.csv')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    main()
