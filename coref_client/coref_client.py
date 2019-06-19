import os
from functools import reduce
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from extract_markable_features import extract_markable_features, save_markable_features
from extract_mention_pair_features import extract_mention_pair_features, save_mention_pair_features
from utils.clusterers import get_anaphora_scores_by_antecedent, BestFirstClusterer
from utils.data_helper import get_markable_dataframe, get_embedding_variables
from utils.training_instances_generator import BudiInstancesGenerator

root_path = os.path.dirname(os.path.abspath(__file__)) + '/../'


class CorefClient:
    def __init__(self, embedding_indexes_file_path: str = root_path + 'helper_files/embedding/embedding_indexes.txt',
                 indexed_embedding_file_path: str = root_path + 'helper_files/embedding/indexed_embedding.txt',
                 singleton_classifier_file_path: str = root_path +
                                                       'models/singleton_classifiers/words_context_syntactic.model',
                 coreference_classifier_file_path: str = root_path + 'models/coreference_classifiers/'
                                                                     'proposed_hyperparameter/'
                                                                     'words_context_syntactic_budi_20.model',
                 singleton_classifier_threshold: float = 0.4,
                 coreference_classifier_threshold: float = 0.2,
                 max_text_length: int = 10,
                 max_prev_words_length: int = 10,
                 max_next_words_length: int = 10):
        self.word_vector, self.embedding_matrix, self.idx_by_word, self.word_by_idx = get_embedding_variables(
            embedding_indexes_file_path,
            indexed_embedding_file_path)
        self.singleton_classifier: Model = load_model(singleton_classifier_file_path)
        self.coreference_classifier: Model = load_model(coreference_classifier_file_path)
        self.singleton_classifier_threshold = singleton_classifier_threshold
        self.coreference_classifier_threshold = coreference_classifier_threshold
        self.max_text_length = max_text_length
        self.max_prev_words_length = max_prev_words_length
        self.max_next_words_length = max_next_words_length

        self.graph = tf.get_default_graph()

    def get_markable_clusters(self, data_file_path: str, use_singleton_classifier: bool) -> List[List[int]]:
        mention_pair_prediction = self.get_mention_pair_prediction(data_file_path, use_singleton_classifier)
        markable_clusters = BestFirstClusterer().get_chains(mention_pair_prediction,
                                                            self.coreference_classifier_threshold)
        return markable_clusters

    def get_mention_pair_prediction(self, data_file_path: str, use_singleton_classifier: bool) -> \
            Dict[int, List[Tuple[int, float]]]:
        markables = self.get_markable_data(data_file_path)

        tmp_csv_file_path = data_file_path + '_mention_pair.csv'
        document_id_by_markable_id = {x: 1 for x in range(101)}  # Only one document, max 100 markables

        mention_pairs = extract_mention_pair_features(data_file_path,
                                                      BudiInstancesGenerator(document_id_by_markable_id))
        save_mention_pair_features(mention_pairs, tmp_csv_file_path)

        mention_pair_dataframe = pd.read_csv(tmp_csv_file_path)
        os.remove(tmp_csv_file_path)

        def get_list_of_markable_data(key: str, ids: Iterable) -> List:
            return list(map(lambda idx: markables[idx][key], ids))

        text_1 = get_list_of_markable_data('text', mention_pair_dataframe.m1_id)
        prev_1 = get_list_of_markable_data('previous_words', mention_pair_dataframe.m1_id)
        next_1 = get_list_of_markable_data('next_words', mention_pair_dataframe.m1_id)
        numeric_1 = get_list_of_markable_data('numeric', mention_pair_dataframe.m1_id)

        text_2 = get_list_of_markable_data('text', mention_pair_dataframe.m2_id)
        prev_2 = get_list_of_markable_data('previous_words', mention_pair_dataframe.m2_id)
        next_2 = get_list_of_markable_data('next_words', mention_pair_dataframe.m2_id)
        numeric_2 = get_list_of_markable_data('numeric', mention_pair_dataframe.m2_id)

        relation = mention_pair_dataframe[
            ['is_exact_match', 'is_words_match', 'is_substring', 'is_abbreviation', 'is_appositive',
             'is_nearest_candidate', 'sentence_distance', 'word_distance', 'markable_distance']]

        prediction = self.coreference_classifier.predict(
            [text_1, text_2, prev_1, prev_2, next_1, next_2, numeric_1, numeric_2, relation])

        if use_singleton_classifier:
            singletons = set([idx for idx, markable in markables.items() if markable['is_singleton']])
        else:
            singletons = set()

        anaphora_scores_by_antecedent = get_anaphora_scores_by_antecedent(mention_pair_dataframe.m1_id,
                                                                          mention_pair_dataframe.m2_id,
                                                                          prediction,
                                                                          singletons)

        return anaphora_scores_by_antecedent

    def get_markable_data(self, data_file_path: str) -> Dict[int, Dict]:
        tmp_csv_file_path = data_file_path + '_markable.csv'
        document_id_by_sentence_id = {x: 1 for x in range(21)}  # Only one document, max 20 sentence

        markables = extract_markable_features(data_file_path, document_id_by_sentence_id)
        save_markable_features(markables, tmp_csv_file_path)

        data = get_markable_dataframe(tmp_csv_file_path, self.word_vector, self.idx_by_word)
        os.remove(tmp_csv_file_path)

        data_text = pad_sequences(data.text, maxlen=self.max_text_length, padding='post')
        data_previous_words = pad_sequences(
            data.previous_words.map(lambda seq: seq[(-1 * self.max_prev_words_length):]),
            maxlen=self.max_prev_words_length, padding='pre')
        data_next_words = pad_sequences(data.next_words.map(lambda seq: seq[:self.max_next_words_length]),
                                        maxlen=self.max_next_words_length, padding='post')
        data_numeric = data[['is_pronoun', 'entity_type', 'is_proper_name', 'is_first_person']]

        data_numeric = np.array(list(
            map(lambda p: reduce(lambda x, y: x + y, [i if type(i) is list else [i] for i in p]),
                data_numeric.values)))

        with self.graph.as_default():
            data_is_singleton = self.singleton_classifier.predict(
                [data_text, data_previous_words, data_next_words, data_numeric])

        data_is_singleton = list(
            map(lambda x: 1 if x[1] > self.singleton_classifier_threshold else 0, data_is_singleton))

        return {idx: {'text': text, 'previous_words': previous_words, 'next_words': next_words, 'numeric': numeric,
                      'is_singleton': is_singleton}
                for idx, text, previous_words, next_words, numeric, is_singleton in
                zip(data.id, data_text, data_previous_words, data_next_words, data_numeric, data_is_singleton)}
