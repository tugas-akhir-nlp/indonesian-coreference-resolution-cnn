from typing import Tuple, List, Any


class Rule:
    NAME_CLASSES: List[str] = ['PERSON', 'ORGANIZATION', 'LOCATION', 'UNKNOWN']

    def __init__(self, is_string_match: int, is_string_without_punctuation_match: int, is_abbreviation: int,
                 is_first_pronoun: int, is_second_pronoun: int, is_on_one_sentence: int, is_substring: int,
                 first_name_class: str, second_name_class: str) -> None:
        self.is_string_match = is_string_match
        self.is_string_without_punctuation_match = is_string_without_punctuation_match
        self.is_abbreviation = is_abbreviation
        self.is_first_pronoun = is_first_pronoun
        self.is_second_pronoun = is_second_pronoun
        self.is_on_one_sentence = is_on_one_sentence
        self.is_substring = is_substring
        self.first_name_class = self.NAME_CLASSES.index(first_name_class)
        self.second_name_class = self.NAME_CLASSES.index(second_name_class)

    def get_attribute(self, attribute_name: str) -> Any:
        if attribute_name in ['first_name_class', 'second_name_class']:
            return self.NAME_CLASSES[getattr(self, attribute_name)]

        return getattr(self, attribute_name)

    def __eq__(self, other: 'Rule') -> bool:
        return self.__class__ == other.__class__ \
               and self._get_binary_attributes() == other._get_binary_attributes() \
               and self._get_name_class_attributes() == other._get_name_class_attributes()

    def __hash__(self) -> int:
        binary_attributes = self._get_binary_attributes()
        name_class_attributes = self._get_name_class_attributes()

        hash_value = 0
        for binary_attribute in binary_attributes:
            hash_value = hash_value * 2 + binary_attribute

        for name_class_attribute in name_class_attributes:
            hash_value = hash_value * len(self.NAME_CLASSES) + name_class_attribute

        return hash_value

    def _get_binary_attributes(self) -> Tuple[int, ...]:
        return (
            self.is_string_match,
            self.is_string_without_punctuation_match,
            self.is_abbreviation,
            self.is_first_pronoun,
            self.is_second_pronoun,
            self.is_on_one_sentence,
            self.is_substring
        )

    def _get_name_class_attributes(self) -> Tuple[int, ...]:
        return (
            self.first_name_class,
            self.second_name_class
        )
