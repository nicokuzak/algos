import os
import pandas as pd

from modeling.supervised.data_cleaning import Text
from modeling.supervised.model import NeuralNetwork

path = '/Users/nkuzak003/Documents/data/personal/sentiment labelled sentences/'

df_list = []
for x in os.listdir(path):
    if x[x.rfind('_')+1: x.find('.')] == 'labelled':
        df = pd.read_csv(os.path.join(path, x), names=['sentence', 'label'], sep='\t')
        df['source'] = x[:x.find('_')] 
        df_list.append(df)
df = pd.concat(df_list)

ds = Text(df=df, feature_col='sentence', label_col='label')
save_path = '/Users/nkuzak003/Documents/personal/notebooks/test_data'

ds.train_fasttext(save_path)

model = NeuralNetwork(ds)
layers = [30, 10]
model.embedding_to_sequential(layers)
model.train( validation_split=.2, epochs=10)
model.evaluate()