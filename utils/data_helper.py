from string import punctuation
from typing import Dict, List, Tuple, Callable
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from .stem import IndonesianStemmer
from .ufds import UFDS
import xml.etree.ElementTree as ET

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
        -> Tuple[List[ET.Element], Dict[int, ET.Element], Dict[int,int]]:
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
                phrase_id_by_node_id[int(phrase.attrib['id'])] = len(phrases) - 1

            if 'coref' in phrase.attrib:
                ufds.gabung(int(phrase.attrib['id']), int(phrase.attrib['coref']))

    return phrases, nodes, phrase_id_by_node_id


def get_entity_types(labels: List[str]) -> List[str]:
    entity_types = set()

    for label in labels:
        for entity_type in label.split('|'):
            entity_types.add(entity_type)

    return list(entity_types)


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

    markables.text = markables.text.fillna("").map(lambda x: to_sequence(clean_sentence(str(x), word_vector),
                                                                         idx_by_word))
    markables.is_pronoun = markables.is_pronoun.map(int)
    markables.entity_type = markables.entity_type.map(entity_to_bow(get_entity_types(markables.entity_type)))
    markables.is_proper_name = markables.is_proper_name.map(int)
    markables.is_first_person = markables.is_first_person.map(int)
    markables.previous_words = markables.previous_words.fillna("").map(
        lambda x: to_sequence(clean_sentence(str(x), word_vector), idx_by_word)
    )
    markables.next_words = markables.next_words.fillna("").map(
        lambda x: to_sequence(clean_sentence(str(x), word_vector), idx_by_word)
    )
    markables.is_singleton = markables.is_singleton.map(lambda x: to_categorical(x, num_classes=2))

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
