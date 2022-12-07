#!/usr/bin/env python

import csv
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import contractions
import string
from nltk.corpus import stopwords

class RedditNewsDJIADataset(Dataset):

    def __init__(self, csvFile, train, trainSplit=0.8, transform=None):
        self.train = train
        self.trainSplit = trainSplit
        self.transform = transform
        self.dataFrame = pd.read_csv(csvFile)
        trainSize = int(self.dataFrame.shape[0] * self.trainSplit)

        if self.train:
            self.dataFrame = self.dataFrame[:trainSize]
        else:
            self.dataFrame = self.dataFrame[trainSize:]

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        label = self.dataFrame.Label.values[index]
        features = self.dataFrame.iloc[index,2:].values
        features = np.asarray(features,dtype=str)
        sample = {'label': label, 'features': features}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        label, features = sample['label'], sample['features']

        return {'label': label,
                'features': torch.from_numpy(features)}


class CleanRedditNewsDJIA:

    def __init__(self, fileName: str):
        self.fileName = fileName

    def cleanUp(self):

        with open(self.fileName) as f:
            reader = csv.reader(f)
            formattedDoc = []

            for row in reader:
                formattedRow = []

                for feature in row:
                    feature = feature.replace('\r','')
                    feature = feature.replace('\n','')
                    feature = feature.replace('\t','')
                    feature = feature.replace('\\r','')
                    feature = feature.replace('\\n','')
                    feature = feature.replace('\\t','')
                    feature = feature.replace('\\','')
                    feature = feature.strip('\'\"b ')
                    feature = feature.strip('\'\"')
                    formattedRow.append(feature)

                formattedDoc.append(formattedRow)

        with open(self.fileName[:-4] + '_clean' + self.fileName[-4:], 'w', newline='') as f:
            writer = csv.writer(f)
            for row in formattedDoc:
                writer.writerow(row)


class CleanData:

    def __init__(self, fileName: str, cleanColumns: list):
        self.fileName = fileName
        self.cleanColumns = cleanColumns

    def cleanUp(self):
        csv.field_size_limit(int(sys.maxsize/10000000000))
        print(int(sys.maxsize/10000000000))
        sw_nltk = []
        sw_nltk = stopwords.words('english')
        #print(sw_nltk)
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
                        #feature = contractions.fix(feature)
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
        
        