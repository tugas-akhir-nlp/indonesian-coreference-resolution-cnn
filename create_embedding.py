import numpy as np
from gensim.models import Word2Vec
import xml.etree.ElementTree as ET
from utils.data_helper import clean_sentence

np.random.seed(26061997)

word_vector = Word2Vec.load('helper_files/word2vec/id.bin')
data = ET.parse('data/full.xml')
embedding_file_path = 'helper_files/generated_embedding.txt'

min_val = min(map(min, word_vector.wv.vectors))
max_val = max(map(max, word_vector.wv.vectors))

vector_size = word_vector.wv.vector_size

embedding_file = open(embedding_file_path, 'w')
embedding = {}

root = data.getroot()

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

embedding_file.close()
