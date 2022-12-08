#! /usr/bin/env python3

import csv
import sys
import contractions
import string
from nltk.corpus import stopwords


# CleanData Class
'''
    This class can be used to clean a dataset from a CSV file. 

    fileName: string path to the dataset
    cleanColumns: a list of which columns to run the cleaning on. Ex: [0,3,7]
                    Cleaning should not be ran on all columns such as Date, Time, Non-text, etc. 

    If text was read in as bytes in python, the text will contain extra escape characters
    such as \r, \n, \t, etc. Escape chars are removed. Contractions are broken out into words.
    $ symbols are replaced with the word 'money'. Punctuation is removed. Case is lowered. 
    Stop words are removed using NLTK's library

'''
class CleanData:

    def __init__(self, fileName: str, cleanColumns: list):
        self.fileName = fileName
        self.cleanColumns = cleanColumns

    def cleanUp(self):
        # needed to increase max column size for large documents in csv
        csv.field_size_limit(int(sys.maxsize/10000000000))

        sw_nltk = []
        sw_nltk = stopwords.words('english')

        if 'not' in sw_nltk:
            sw_nltk.remove('not')

        with open(self.fileName, encoding="utf-8") as f:
            reader = csv.reader(f)
            formattedDoc = []

            for row in reader:
                formattedRow = []

                for column, feature in enumerate(row):
                    if column in self.cleanColumns:
                        feature = contractions.fix(feature)
                        feature = feature.replace('\r','')
                        feature = feature.replace('\n','')
                        feature = feature.replace('\t','')
                        feature = feature.replace('\\r','')
                        feature = feature.replace('\\n','')
                        feature = feature.replace('\\t','')
                        feature = feature.replace('\\','')
                        feature = feature.strip('\'\"b ')
                        feature = feature.strip('\'\"')
                        feature = feature.lower()
                        feature = feature.replace('$',' money ')
                        spaces = str(' ' * len(string.punctuation))
                        feature = feature.translate(str.maketrans(string.punctuation,spaces))
                        feature = ' '.join(feature.split())
                        feature = ' '.join([word for word in feature.split() if not word in sw_nltk])

                    formattedRow.append(feature)

                formattedDoc.append(formattedRow)

        with open(self.fileName[:-4] + '_clean' + self.fileName[-4:], 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for row in formattedDoc:
                writer.writerow(row)
        
        