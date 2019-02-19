import numpy as np
from gensim.models import Word2Vec
import xml.etree.ElementTree as ET
from utils.data_helper import clean_sentence

np.random.seed(26061997)

word_vector = Word2Vec.load('helper_files/word2vec/id.bin')
data = ET.parse('data/full.xml')
embedding_file_path = 'helper_files/embedding/generated_embedding.txt'
indexed_embedding_file_path = 'helper_files/embedding/indexed_embedding.txt'
embedding_indexes_file_path = 'helper_files/embedding/embedding_indexes.txt'

min_val = min(map(min, word_vector.wv.vectors))
max_val = max(map(max, word_vector.wv.vectors))

vector_size = word_vector.wv.vector_size

embedding_file = open(embedding_file_path, 'w')
indexed_embedding_file = open(indexed_embedding_file_path, 'w')
embedding_indexes_file = open(embedding_indexes_file_path, 'w')
embedding = {}

root = data.getroot()


idx = 0
for sentence in root:
    for phrase in sentence:
        cleaned_phrase = clean_sentence(phrase.text, word_vector.wv)

        for word in cleaned_phrase.split():
            if word not in embedding:
                if word in word_vector.wv:
                    embedding[word] = word_vector.wv[word]
                else:
                    embedding[word] = np.random.rand(
                        vector_size) * (max_val - min_val) + min_val

                embedding_file.write(
                    word + ' ' + ' '.join(map(str, embedding[word])) + '\n')
                indexed_embedding_file.write(' '.join(map(str, embedding[word])) + '\n')
                embedding_indexes_file.write(word + ' ' + str(idx) + '\n')

                idx += 1

embedding_file.close()
indexed_embedding_file.close()
embedding_indexes_file.close()
