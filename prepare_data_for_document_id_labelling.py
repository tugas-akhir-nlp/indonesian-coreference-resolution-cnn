import xml.etree.ElementTree as ET
from csv import DictWriter

from utils.data_helper import get_words_only

data = ET.parse('data/full.xml')
labelling_file_path = 'data/document_id.csv'

labelling_file = open(labelling_file_path, 'w')
labelling_csv = DictWriter(labelling_file, fieldnames=['sentence_id', 'text', 'document_id'])
labelling_csv.writeheader()

root = data.getroot()

for sentence in root:
    text = ' '.join([get_words_only(phrase.text) for phrase in sentence])
    labelling_csv.writerow({'sentence_id': sentence.attrib['id'], 'text': text, 'document_id': ''})

labelling_file.close()
