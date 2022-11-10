#!/usr/bin/env python

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import csv

class RedditNewsDJIADataset(Dataset):

    def __init__(self, csvFile, train, trainSplit=0.8, transform=None):
        self.train = train
        self.trainSplit = trainSplit
        self.transform = transform
        #split = self.trainSplit * 

        if self.train:
            self.dataFrame = pd.read_csv(csvFile)

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        label = self.dataFrame.Label.values[index]
        features = self.dataFrame.iloc[index,2:].values
        features = np.char.strip(np.asarray(features,dtype=str),chars='b\'\"')
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

        #l = self.fileName[:-4] + '_clean' + self.fileName[-4:]
        print('test',chr(10),'test')
        
        formatted = []
        flag = True
        with open(self.fileName) as f:
            reader = csv.reader(f)
            for row in reader:
                index = 0
                flag = True
                while flag:
                    try:
                        index = row.index(chr(10),index)
                        print(ord(row[index-1]), ord(row[index]))
                        if row[index-1] != chr(13):
                            del row[index]
                            index -= 1
                        else:
                            index += 1
                    except (ValueError, IndexError) as err:
                        flag = False
                        formatted.append(row)
        
        with open(self.fileName[:-4] + '_clean' + self.fileName[-4:], 'w', newline='') as f:
            writer = csv.writer(f)
            for row in formatted:
                writer.writerow(row)
        