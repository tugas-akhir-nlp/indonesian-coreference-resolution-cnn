from csv import DictWriter, DictReader
from typing import Dict, Tuple

from baseline.budi_et_al.rule import Rule


class Model:
    freeze: bool = False
    count: int = 0
    rule_counter: Dict[Rule, Tuple[int, int]] = {}
    rule_support: Dict[Rule, float] = {}
    rule_confidence: Dict[Rule, float] = {}

    def predict(self, rule: Rule, threshold: float = 0.5) -> int:
        if rule not in self.rule_confidence:
            return 0

        return int(self.rule_confidence[rule] > threshold)

    def add_rule(self, rule: Rule, is_coreference: int) -> None:
        if self.freeze:
            raise Exception('Model has been trained, rule addition is not allowed')

        if rule in self.rule_counter:
            current_count = self.rule_counter[rule]
        else:
            current_count = (0, 0)

        current_count = (current_count[0] + 1 - is_coreference, current_count[1] + is_coreference)

        self.rule_counter[rule] = current_count
        self.count += 1

    def train(self) -> None:
        for rule in self.rule_counter.keys():
            negatives, positives = self.rule_counter[rule]

            if self.count > 0:
                self.rule_support[rule] = (negatives + positives) / self.count
            else:
                self.rule_support[rule] = 0

            if positives + negatives > 0:
                self.rule_confidence[rule] = positives / (positives + negatives)
            else:
                self.rule_confidence[rule] = 0

        self.freeze = True

    def get_variables(self) -> Tuple[Dict[Rule, float], Dict[Rule, float]]:
        if not self.freeze:
            self.train()

        return self.rule_support, self.rule_confidence

    def save(self, path: str) -> None:
        rule_fields = ['is_string_match', 'is_string_without_punctuation_match', 'is_abbreviation',
                       'is_first_pronoun', 'is_second_pronoun', 'is_on_one_sentence', 'is_substring',
                       'first_name_class', 'second_name_class']

        fields = rule_fields + ['negative', 'positive', 'support', 'confidence']

        if not self.freeze:
            self.train()

        with open(path, 'w') as f:
            csv_file = DictWriter(f, fieldnames=fields)
            csv_file.writeheader()

            for rule in self.rule_counter.keys():
                rule_dict = {}
                for field in rule_fields:
                    rule_dict[field] = rule.get_attribute(field)

                rule_dict['negative'], rule_dict['positive'] = self.rule_counter[rule]
                rule_dict['support'] = self.rule_support[rule]
                rule_dict['confidence'] = self.rule_confidence[rule]

                csv_file.writerow(rule_dict)

    @staticmethod
    def load(path: str) -> 'Model':
        model = Model()

        rule_fields = ['is_string_match', 'is_string_without_punctuation_match', 'is_abbreviation',
                       'is_first_pronoun', 'is_second_pronoun', 'is_on_one_sentence', 'is_substring',
                       'first_name_class', 'second_name_class']

        with open(path, 'r') as f:
            csv_file = DictReader(f)

            for rule_dict in csv_file:
                rule = Rule(**{field: rule_dict[field] for field in rule_fields})
                model.rule_counter[rule] = (rule_dict['negative'], rule_dict['positive'])
                model.rule_support[rule] = rule_dict['support']
                model.rule_confidence[rule] = rule_dict['confidence']
                model.count += rule_dict['negative'] + rule_dict['positive']

        model.freeze = True
        return model
