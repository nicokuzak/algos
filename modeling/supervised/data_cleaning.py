import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:

    def __init__(self, file=None, df=None, label_col=''):
        assert file is not None or df is not None

        if file is not None:
            if file[file.rfind('.'):] == 'csv':
                self.df = pd.read_csv(file)
            elif file[file.rfind('.'):] == 'xlsx':
                self.df = pd.read_excel(file)
            else:
                raise ValueError(f'Cannot recognize the {file} extension')
        
        if df is not None:
            self.df = df
        if label_col == '' or label_col not in df.columns:
            raise ValueError(f'{label_col} not in data.')
        
        self.label_col = label_col
        self.features = [col for col in df.columns if col != label_col]
        self.changed = {}
        self.refresh()

    def one_hot_encode(self, col):
        new = pd.get_dummies(self.df[col], prefix=col)
        self.df = pd.concat([self.df, new], axis=1)
        self.features = [x for x in self.features if x != col] + [x for x in new.columns()]
        self.refresh()
    
    def to_numerical(self, col, val):
        vals = self.df[col].unique() 
        assert len(vals) == 2
        self.df[col] = pd.Series(np.where(self.df[col] == val, 1, 0), name = col)
        non_val = [x for x in vals if x != val]
        self.changed[col] = {val:1, non_val:0}
    
    def fill_missing(self, col, how=''):
        fill = {'': '', 
        'mean':self.df[col].mean(),
        'mode':self.df[col].mode()[0]
        }
        if how in fill.keys():
            self.df[col] = self.df[col].fillna(fill[how])
        else:
            self.df[col] = self.df[col].fillna(how)
        
    
    def refresh(self):
        self.X = self.df[self.features]
        self.y = self.df[self.label_col]

    def split(self):
        self.refresh()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)