from numpy.random import RandomState
import numpy as np
import pandas as pd

df = pd.read_csv("/Users/macintosh/Desktop/TitleAnalyzer/data/book30-listing-test.csv", encoding = "ISO-8859-1", header=None)
print(df.shape)
rng = RandomState()

train = df.sample(frac=0.5, random_state=rng)
test = df.loc[~df.index.isin(train.index)]

print(train.shape)
print(test.shape)
train.to_csv('/Users/macintosh/Desktop/TitleAnalyzer/data/split-2850/train.csv')
test.to_csv('/Users/macintosh/Desktop/TitleAnalyzer/data/split-2850/test.csv')

data1 = pd.read_csv('/Users/macintosh/Desktop/TitleAnalyzer/data/split-2850/train.csv', encoding = "ISO-8859-1")
data2 = pd.read_csv('/Users/macintosh/Desktop/TitleAnalyzer/data/split-2850/train.csv', encoding = "ISO-8859-1")
print(data1)
print(data1.shape)
print(data2.shape)

print('Files saved')