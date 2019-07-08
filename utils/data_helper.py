import xml.etree.ElementTree as ET
from csv import DictReader
from string import punctuation
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

from .data_structures import UFDS
from .stem import IndonesianStemmer

stemmer = IndonesianStemmer()


def is_number(word: str) -> bool:
    return word.replace(',', '.').replace('.', '').replace('-', '', 1).isdigit()


def get_word(word: str) -> str:
    if '\\' in word:
        word = word.split('\\')[0]

    while word[-1] in punctuation and len(word) > 1:
        word = word[:-1]

    while word[0] in punctuation and len(word) > 1:
        word = word[1:]

    word = word.lower()

    return word


def get_words_only(text: str) -> str:
    words = text.split()
    words = map(get_word, words)
    return ' '.join(words)


def get_abbreviation(text: str) -> str:
    words = get_words_only(text).split()
    abb = ''

    for word in words:
        abb += word[0]

    return abb


def clean_word(word: str, word_vector: Dict[str, np.array]) -> str:
    word = get_word(word)

    if word not in word_vector:
        tmp = word.split('-')
        if len(tmp) == 2 and tmp[0] == tmp[1]:
            word = tmp[0]

    if word not in word_vector:
        word = stemmer.stem(word)

    if word not in word_vector:
        tmp = word.split('-')
        if len(tmp) == 2 and tmp[0] == tmp[1]:
            word = tmp[0]

    if word not in word_vector:
        word = stemmer.stem(word)

    if is_number(word):
        word = '<angka>'

    return word


def clean_sentence(sentence: str, word_vector: Dict[str, np.array]) -> str:
    return ' '.join([clean_word(word, word_vector) for word in sentence.split() if clean_word(word, word_vector) != ''])


def clean_arr(arr: List[str], word_vector: Dict[str, np.array]) -> List[str]:
    return [clean_word(word, word_vector) for word in arr if clean_word(word, word_vector) != '']


def get_phrases_and_nodes(ufds: UFDS, root_element: ET.Element) \
        -> Tuple[List[ET.Element], Dict[int, ET.Element], Dict[int, int]]:
    # ret: phrases, nodes, phrase_id_by_node_id

    phrases = []
    nodes = {}
    phrase_id_by_node_id = {}

    for sentence in root_element:
        for phrase in sentence:
            phrases.append(phrase)

            if 'id' in phrase.attrib:
                ufds.init_id(int(phrase.attrib['id']))
                nodes[int(phrase.attrib['id'])] = phrase
                phrase_id_by_node_id[int(phrase.attrib['id'])] = len(
                    phrases) - 1

            if 'coref' in phrase.attrib:
                ufds.join(int(phrase.attrib['id']),
                          int(phrase.attrib['coref']))

    return phrases, nodes, phrase_id_by_node_id


def get_entity_types(labels: List[str]) -> List[str]:
    entity_types = set()

    for label in labels:
        for entity_type in label.split('|'):
            entity_types.add(entity_type)

    return sorted(list(entity_types))


def entity_to_bow(entities: List[str]) -> Callable[[str], List[int]]:
    idx = {entities[i]: i for i in range(len(entities))}

    def f(label: str) -> List[int]:
        bow = [0 for _ in entities]

        for entity_type in label.split('|'):
            bow[idx[entity_type]] = 1

        return bow

    return f


def entity_to_id(entities: List[str]) -> Callable[[str], int]:
    idx = {entities[i]: i for i in range(len(entities))}

    def f(label: str) -> int:
        return idx[label]

    return f


def to_sequence(text: str, idx_by_word: Dict[str, int]) -> List[int]:
    text = text.split()
    return list(map(lambda word: idx_by_word[word], text))


def get_markable_dataframe(markable_file: str, word_vector: Dict[str, np.array],
                           idx_by_word: Dict[str, int]) -> pd.DataFrame:
    markables = pd.read_csv(markable_file)
    print(get_entity_types(markables.entity_type))
    markables.text = markables.text.fillna("").map(lambda x: to_sequence(clean_sentence(str(x), word_vector),
                                                                         idx_by_word))
    markables.is_pronoun = markables.is_pronoun.map(int)
    # markables.entity_type = markables.entity_type.map(entity_to_bow(get_entity_types(markables.entity_type)))
    markables.entity_type = markables.entity_type.map(entity_to_bow(
        ['EVENT', 'FACILITY', 'LOCATION', 'NUM', 'ORGANIZATION', 'OTHER', 'PERSON', 'THINGS', 'TIME', 'TITLE']))
    markables.is_proper_name = markables.is_proper_name.map(int)
    markables.is_first_person = markables.is_first_person.map(int)
    markables.previous_words = markables.previous_words.fillna("").map(
        lambda x: to_sequence(clean_sentence(str(x), word_vector), idx_by_word)
    )
    markables.next_words = markables.next_words.fillna("").map(
        lambda x: to_sequence(clean_sentence(str(x), word_vector), idx_by_word)
    )
    markables.all_previous_words = markables.all_previous_words.fillna("").map(
        lambda x: to_sequence(clean_sentence(str(x), word_vector), idx_by_word)
    )
    markables.is_singleton = markables.is_singleton.map(
        lambda x: to_categorical(x, num_classes=2))

    return markables


def get_embedding_variables(embedding_indexes_file_path: str,
                            indexed_embedding_file_path: str) \
        -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, int], Dict[int, str]]:
    word_vector = {}
    embedding_matrix = []
    idx_by_word = {}
    word_by_idx = {}

    for element in open(embedding_indexes_file_path, 'r').readlines():
        element = element.split()
        word, index = element[0], int(element[1])
        idx_by_word[word] = index
        word_by_idx[index] = word

    for element in open(indexed_embedding_file_path, 'r').readlines():
        element = element.split()

        embedding = np.asarray(element, dtype='float64')

        if len(embedding_matrix) == 0:
            embedding_matrix.append(np.zeros(embedding.shape))

        index = len(embedding_matrix)
        word_vector[word_by_idx[index]] = embedding

        embedding_matrix.append(embedding)

    embedding_matrix = np.array(embedding_matrix)

    return word_vector, embedding_matrix, idx_by_word, word_by_idx


def get_sentence_variables(data_file_path: str) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    data = ET.parse(data_file_path)
    root = data.getroot()

    sentence_id_by_markable_id = {}
    markable_ids_by_sentence_id = {}

    for sentence in root:
        markable_ids_by_sentence_id[int(sentence.attrib['id'])] = []

        for phrase in sentence:
            if 'id' in phrase.attrib:
                sentence_id_by_markable_id[int(phrase.attrib['id'])] = int(
                    sentence.attrib['id'])
                markable_ids_by_sentence_id[int(sentence.attrib['id'])].append(
                    int(phrase.attrib['id']))

    return sentence_id_by_markable_id, markable_ids_by_sentence_id


def get_document_id_variables(document_id_file_path: str, markable_ids_by_sentence_id: Dict[int, List[int]]) \
        -> Tuple[Dict[int, int], Dict[int, int], Dict[int, List[int]], Dict[int, List[int]]]:
    document_id_by_sentence_id = {}
    document_id_by_markable_id = {}
    sentence_ids_by_document_id = {}
    markable_ids_by_document_id = {}

    with open(document_id_file_path, 'r') as file:
        csv_file = DictReader(file)

        for row in csv_file:
            sentence_id = int(row['sentence_id'])
            document_id = int(row['document_id'])

            if document_id not in markable_ids_by_document_id:
                markable_ids_by_document_id[document_id] = []

            if document_id not in sentence_ids_by_document_id:
                sentence_ids_by_document_id[document_id] = []

            document_id_by_sentence_id[sentence_id] = document_id
            sentence_ids_by_document_id[document_id].append(sentence_id)

            for markable_id in markable_ids_by_sentence_id[sentence_id]:
                document_id_by_markable_id[markable_id] = document_id
                markable_ids_by_document_id[document_id].append(markable_id)

    return (document_id_by_sentence_id, document_id_by_markable_id, sentence_ids_by_document_id,
            markable_ids_by_document_id)
