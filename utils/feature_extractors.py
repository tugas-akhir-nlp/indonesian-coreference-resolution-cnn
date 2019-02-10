from xml.etree.ElementTree import Element
from .data_helper import get_words_only, get_word, get_abbreviation
from typing import List, Dict


class FeatureExtractor:
    features = []

    def get_features(self, *args, **kwargs)-> dict:
        features = {}

        for feature in self.features:
            if not hasattr(self, 'get_' + feature):
                raise Exception('Feature %s not implemented' % feature)

            feature_getter = getattr(self, 'get_' + feature)

            if not callable(feature_getter):
                raise Exception('get_%s should be callable' % feature)

            features[feature] = feature_getter(*args, **kwargs)

        return features


class SingleSyntacticFeatureExtractor(FeatureExtractor):
    features = ['is_pronoun', 'entity_type', 'is_proper_name', 'is_first_person', 'num_words', 'previous_words',
                'next_words']
    # should we add is in quotation?

    def __init__(self, phrases: List[Element], nodes: Dict[int, Element], phrase_id_by_node_id: Dict[int, int],
                 context_words_count=10) -> None:
        self.phrases = phrases
        self.nodes = nodes
        self.phrase_id_by_node_id = phrase_id_by_node_id
        self.context_words_count = context_words_count

    def get_entity_type(self, node: Element) -> str:
        if self.get_is_pronoun(node):
            return 'PERSON'

        return node.attrib['ne']

    def get_is_pronoun(self, node: Element) -> bool:
        pronouns = ['dia', 'ia', 'aku', 'saya', 'kamu', 'engkau', 'kau', 'anda', 'kami', 'kalian', 'kita', 'mereka',
                    'beliau']

        if '\\PRP' in node.text:
            return True

        phrase = get_words_only(node.text)

        return phrase in pronouns

    def get_is_proper_name(self, node: Element) -> bool:
        return '\\NNP' in node.text

    def get_is_first_person(self, node: Element) -> bool:
        first_persons = ['saya', 'aku']
        phrase = get_words_only(node.text)

        return phrase in first_persons

    def get_num_words(self, node: Element) -> int:
        return len(node.text.split())

    def get_previous_words(self, node: Element) -> str:
        phrase_id = self.phrase_id_by_node_id[int(node.attrib['id'])]

        current_phrase_id = phrase_id - 1
        words_left = self.context_words_count

        previous_words = ''

        while words_left > 0 and current_phrase_id >= 0:
            words = self.phrases[current_phrase_id].text.split()

            if len(words) <= words_left:
                previous_words = ' '.join(words) + previous_words
                words_left -= len(words)
                current_phrase_id -= 1
            else:
                previous_words = ' '.join(words[(-1*words_left):]) + previous_words
                words_left = 0

        return previous_words

    def get_next_words(self, node: Element) -> str:
        phrase_id = self.phrase_id_by_node_id[int(node.attrib['id'])]

        current_phrase_id = phrase_id + 1
        words_left = self.context_words_count

        next_words = ''

        while words_left > 0 and current_phrase_id < len(self.phrases):
            words = self.phrases[current_phrase_id].text.split()

            if len(words) <= words_left:
                next_words = ' '.join(words) + next_words
                words_left -= len(words)
                current_phrase_id += 1
            else:
                next_words = ' '.join(words[:words_left]) + next_words
                words_left = 0

        return next_words


class PairSyntacticFeatureExtractor(FeatureExtractor):
    features = ['is_exact_match', 'is_words_match', 'is_substring', 'is_abbreviation', 'is_appositive',
                'is_nearest_candidate', 'sentence_distance', 'word_distance', 'markable_distance']

    def __init__(self, parent_map: Dict[Element, Element], phrases: List[Element],
                 phrase_id_by_node_id: Dict[int, int]) -> None:
        self.parent_map = parent_map
        self.phrases = phrases
        self.phrase_id_by_node_id = phrase_id_by_node_id

    def get_is_exact_match(self, node1: Element, node2: Element) -> bool:
        phrase1 = get_words_only(node1.text)
        phrase2 = get_words_only(node2.text)

        return phrase1 == phrase2

    def get_is_words_match(self, node1: Element, node2: Element) -> bool:
        words1 = list(map(get_word, node1.text.split()))
        words2 = list(map(get_word, node2.text.split()))

        for word in words1:
            if word not in words2:
                return False

        for word in words2:
            if word not in words1:
                return False

        return True

    def get_is_substring(self, node1: Element, node2: Element) -> bool:
        phrase1 = get_words_only(node1.text)
        phrase2 = get_words_only(node2.text)

        return (phrase1 in phrase2) or (phrase2 in phrase1)

    def get_is_abbreviation(self, node1: Element, node2: Element) -> bool:
        return get_abbreviation(node1.text) == get_words_only(node2.text) or \
               get_abbreviation(node2.text) == get_words_only(node1.text)

    def get_is_appositive(self, node1: Element, node2: Element) -> bool:
        first_node = node1 if int(node1.attrib['id']) < int(node2.attrib['id']) else node2
        return self.get_is_nearest_candidate(node1, node2) and first_node.text.split('\\')[-2][-1] == ','

    def get_is_nearest_candidate(self, node1: Element, node2: Element) -> bool:
        return abs(int(node2.attrib['id']) - int(node1.attrib['id'])) == 1
    
    def get_sentence_distance(self, node1: Element, node2: Element) -> bool:
        return abs(int(self.parent_map[node1].attrib['id']) - int(self.parent_map[node2].attrib['id']))

    def get_word_distance(self, node1: Element, node2: Element) -> bool:
        phrase_id_1 = self.phrase_id_by_node_id[int(node1.attrib['id'])]
        phrase_id_2 = self.phrase_id_by_node_id[int(node2.attrib['id'])]

        dist = 0
        for phrase_id in range(min(phrase_id_1, phrase_id_2) + 1, max(phrase_id_1, phrase_id_2)):
            dist += self.phrases[phrase_id]

        return dist

    def get_markable_distance(self, node1: Element, node2: Element) -> bool:
        return abs(int(node1.attrib['id']) - int(node2.attrib['id']))