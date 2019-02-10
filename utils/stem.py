# -*- coding: utf-8 -*-
# source: https://github.com/bogdan-ivanov/IndonesianStemmer/
from nltk.stem.snowball import StemmerI


class IndonesianStemmer(StemmerI):
    """
    The indonesian snowball stemmer
    """

    REMOVED_KE = 1
    REMOVED_PENG = 2
    REMOVED_DI = 4
    REMOVED_MENG = 8
    REMOVED_TER = 16
    REMOVED_BER = 32
    REMOVED_PE = 64

    def __init__(self):
        self.num_syllables = 0
        self.stem_derivational = True

    def stem(self, word):
        self.flags = 0
        # number of syllables == number of vowels
        self.num_syllables = len([c for c in word if self._is_vowel(c)])
        self.word = word

        if self.num_syllables > 2:
            self._remove_particle()

        if self.num_syllables >= 2:
            self._remove_possessive_pronoun()

        if self.stem_derivational:
            self._stem_derivational()

        return self.word

    def _is_vowel(self, letter):
        """
        Vowels in indonesian
        """
        return letter in [u'a', u'e', u'i', u'o', u'u']

    def _remove_particle(self):
        """
        Remove common indonesian particles, adjust number of syllables
        """
        suffix = self.word[-3:]
        if suffix in [u'kah', u'lah', u'pun']:
            self.num_syllables -= 1
            self.word = self.word[:-3]

    def _remove_possessive_pronoun(self):
        """
        Remove possessive pronoun particles
        """
        if self.word[-2:] in [u'ku', u'mu']:
            self.num_syllables -= 2
            self.word = self.word[:-2]
            return

        if self.word[-3:] == u'nya':
            self.num_syllables -= 1
            self.word = self.word[:-3]

    def _stem_derivational(self):
        old_length = len(self.word)

        if self.num_syllables > 2:
            self._remove_first_order_prefix()

        if old_length != len(self.word):  # A rule has fired
            old_length = len(self.word)

            if self.num_syllables > 2:
                self._remove_suffix()

            if old_length != len(self.word):  # A rule has fired
                if self.num_syllables > 2:
                    self._remove_second_order_prefix

        else:  # fail
            if self.num_syllables > 2:
                self._remove_second_order_prefix()

            if self.num_syllables > 2:
                self._remove_suffix()

    def _remove_first_order_prefix(self):
        """ Remove FIRST ORDER PREFIX """
        if self.word.startswith(u'meng'):
            self.flags |= IndonesianStemmer.REMOVED_MENG
            self.num_syllables -= 1
            self.word = self.word[4:]
            return

        if self.word.startswith(u'meny') and \
           len(self.word) > 4 and self._is_vowel(self.word[4]):
            self.flags |= IndonesianStemmer.REMOVED_MENG
            self.word = self.word[0:3] + u's' + self.word[4:]
            self.num_syllables -= 1
            self.word = self.word[3:]
            return

        if self.word[0:3] in [u'men', u'mem']:
            self.flags |= IndonesianStemmer.REMOVED_MENG
            self.num_syllables -= 1
            self.word = self.word[3:]
            return

        if self.word.startswith(u'me'):
            self.flags |= IndonesianStemmer.REMOVED_MENG
            self.num_syllables -= 1
            self.word = self.word[2:]
            return

        if self.word.startswith(u'peng'):
            self.flags |= IndonesianStemmer.REMOVED_PENG
            self.num_syllables -= 1
            self.word = self.word[4:]
            return

        if self.word.startswith(u'peny') and \
                len(self.word) > 4 and self._is_vowel(self.word[4]):
            self.flags |= IndonesianStemmer.REMOVED_PENG
            #self.word[3] = u's'
            self.word = self.word[0:3] + u's' + self.word[4:]
            self.num_syllables -= 1
            self.word = self.word[3:]
            return

        if self.word.startswith(u'peny'):
            self.flags |= IndonesianStemmer.REMOVED_PENG
            self.num_syllables -= 1
            self.word = self.word[4:]
            return

        if self.word.startswith(u'pen') and \
           len(self.word) > 3 and self._is_vowel(self.word[3]):
            self.flags |= IndonesianStemmer.REMOVED_PENG
            #self.word[2] = u't'
            self.word = self.word[0:2] + u't' + self.word[3:]
            self.num_syllables -= 1
            self.word = self.word[2:]
            return

        if self.word[0:3] in [u'pen', u'pem']:
            self.flags |= IndonesianStemmer.REMOVED_PENG
            self.num_syllables -= 1
            self.word = self.word[2:]
            return

        if self.word.startswith(u'di'):
            self.flags |= IndonesianStemmer.REMOVED_DI
            self.num_syllables -= 1
            self.word = self.word[2:]
            return

        if self.word.startswith(u'ter'):
            self.flags |= IndonesianStemmer.REMOVED_TER
            self.num_syllables -= 1
            self.word = self.word[3:]
            return

        if self.word.startswith(u'ke'):
            self.flags |= IndonesianStemmer.REMOVED_KE
            self.num_syllables -= 1
            self.word = self.word[2:]
            return

    def _remove_second_order_prefix(self):
        """ Remove SECOND ORDER PREFIX """
        if self.word.startswith(u'ber'):
            self.flags |= IndonesianStemmer.REMOVED_BER
            self.num_syllables -= 1
            self.word = self.word[3:]
            return

        if self.word == u'belajar':
            self.flags |= IndonesianStemmer.REMOVED_BER
            self.num_syllables -= 1
            self.word = self.word[3:]
            return

        if self.word.startswith(u'be') and len(self.word) > 4 and \
           not self._is_vowel(self.word[2]) \
           and self.word[3] == u'e' and self.word[4] == 'r':
            self.flags |= IndonesianStemmer.REMOVED_BER
            self.num_syllables -= 1
            self.word = self.word[2:]
            return

        if self.word.startswith(u'per'):
            self.num_syllables -= 1
            self.word = self.word[3:]
            return

        if self.word == u'pelajar':
            self.num_syllables -= 1
            self.word = self.word[3:]
            return

        if self.word.startswith(u'pe'):
            self.flags |= IndonesianStemmer.REMOVED_PE
            self.num_syllables -= 1
            self.word = self.word[2:]
            return

    def _remove_suffix(self):
        if self.word.endswith(u'kan') \
           and not self.flags & IndonesianStemmer.REMOVED_KE \
           and not self.flags & IndonesianStemmer.REMOVED_PENG \
           and not self.flags & IndonesianStemmer.REMOVED_PE:
            self.num_syllables -= 1
            self.word = self.word[:-3]
            return

        if self.word.endswith(u'an') \
           and not self.flags & IndonesianStemmer.REMOVED_DI \
           and not self.flags & IndonesianStemmer.REMOVED_MENG \
           and not self.flags & IndonesianStemmer.REMOVED_TER:
            self.num_syllables -= 1
            self.word = self.word[:-2]
            return

        if self.word.endswith(u'i') \
           and not self.word.endswith(u'si') \
           and not self.flags & IndonesianStemmer.REMOVED_BER \
           and not self.flags & IndonesianStemmer.REMOVED_KE \
           and not self.flags & IndonesianStemmer.REMOVED_PENG:
            self.num_syllables -= 1
            self.word = self.word[:-1]
            return


if __name__ == "__main__":
    stemmer = IndonesianStemmer()
    assert stemmer.stem(u'diakah') == u'dia'
    assert stemmer.stem(u'sayalah') == u'saya'
    assert stemmer.stem(u'tasmu') == u'tas'
    assert stemmer.stem(u'sepedaku') == u'sepeda'
    assert stemmer.stem(u'berlari') == u'lari'
    assert stemmer.stem(u'dimakan') == u'makan'
    assert stemmer.stem(u'kekasih') == u'kasih'
    assert stemmer.stem(u'mengambil') == u'ambil'
    assert stemmer.stem(u'pengatur') == u'atur'
    assert stemmer.stem(u'perlebar') == u'lebar'
    assert stemmer.stem(u'terbaca') == u'baca'
    assert stemmer.stem(u'gulai') == u'gula'
    assert stemmer.stem(u'makanan') == u'makan'
    assert stemmer.stem(u'permainan') == u'main'
    assert stemmer.stem(u'kemenangan') == u'menang'
    assert stemmer.stem(u'berjatuhan') == u'jatuh'
    assert stemmer.stem(u'mengambili') == u'ambil'
