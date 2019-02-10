from .stem import IndonesianStemmer
from string import punctuation
from typing import Dict, List
import numpy as np

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
