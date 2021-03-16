import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from gensim.models import FastText
from gensim.utils import tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class Dataset:
    def __init__(self, file=None, df=None, features_cols=[], label_col=''):
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
        self.features = features_cols
        if self.features == []:
            self.features = [
                col for col in self.df.columns if col != self.label_col
            ]
        self.changed = {}
        self.refresh()

    def fill_missing(self, col, how=''):
        fill = {
            '': '',
            'mean': self.df[col].mean(),
            'mode': self.df[col].mode()[0]
        }
        if how in fill.keys():
            self.df[col] = self.df[col].fillna(fill[how])
        else:
            self.df[col] = self.df[col].fillna(how)

    def refresh(self):
        self.X = self.df[self.features]
        self.y = self.df[self.label_col]

    def split(self, np=False):
        self.refresh()
        if np:
            X = X.values
            y = y.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y)


class Tabular(Dataset):
    def __init__(self, file=None, df=None, label_col=''):
        super().__init__(file, df, label_col)

    def one_hot_encode(self, col):
        new = pd.get_dummies(self.df[col], prefix=col)
        self.df = pd.concat([self.df, new], axis=1)
        self.features = [x for x in self.features if x != col
                         ] + [x for x in new.columns]
        self.refresh()

    def to_numerical(self, col, val):
        vals = self.df[col].unique()
        assert len(vals) == 2
        self.df[col] = pd.Series(np.where(self.df[col] == val, 1, 0), name=col)
        non_val = [x for x in vals if x != val]
        self.changed[col] = {val: 1, non_val: 0}


class Text(Dataset):
    def __init__(self, file=None, df=None, feature_col='Text', label_col=''):
        super().__init__(file, df, feature_col, label_col)
        self.text = self.X
        self.weights = None
        self.split_text()
        # Split into text train and test

    def split_text(self):
        self.refresh()
        self.sentence_train, self.sentence_test, self.y_train, self.y_test = train_test_split(
            self.text, self.y)

    def bag_of_words(self, **kwargs):
        """Transform text corpus into bag of words
        i.e ['Hi you, how are you', 'I am doing well, thank you!'] -> [[1, 1, 1, 2, 0, 0, 0, 0, 0],  [0, 0, 0, 1, 1, 1, 1, 1, 1]]
        """
        self.vectorizer = CountVectorizer(**kwargs)
        self.vectorizer.fit(self.sentence_train)

        self.BoW_train = self.vectorizer.transform(
            self.sentence_train).toarray()
        self.BoW_test = self.vectorizer.transform(self.sentence_test).toarray()
        self.X_train = self.BoW_train
        self.X_test = self.BoW_test

        self.feature_names = self.vectorizer.get_feature_names()

    def vectorize(self, num_words=10000):
        """Transform text corpus to integers in a tokenizer
        i.e. ["Hi how are you?", "I'm well, how about you"] becomes [[10, 3, 4, 7, 0], [5, 12, 3, 15, 7]]
        """
        self.vectorizer = Tokenizer(num_words)
        self.vectorizer.fit_on_texts(self.sentence_train)

        self.tokenized_train = self.vectorizer.texts_to_sequences(
            self.sentence_train)
        self.tokenized_test = self.vectorizer.texts_to_sequences(
            self.sentence_test)

        self.wtoi = self.vectorizer.word_index
        self.itow = self.vectorizer.index_word
        self.pad_and_refresh()

        
    def pad_and_refresh(self, max_len=None):
        if max_len is None:
            self.tokenized_train = pad_sequences(self.tokenized_train,
                                                padding='post')
            self.tokenized_test = pad_sequences(self.tokenized_test,
                                                padding='post')
        else:
            self.tokenized_train = pad_sequences(self.tokenized_train,
                                                padding='post', max_len=max_len)
            self.tokenized_test = pad_sequences(self.tokenized_test,
                                                padding='post', max_len=max_len)
            
        self.X_train = self.tokenized_train
        self.X_test = self.tokenized_test

        self.vocab_size = len(self.wtoi) + 1

    def create_pretrained_embedding_matrix(self, path, embedding_dim=300):
        # works after vectorize
        self.weights = np.zeros((self.vocab_size, embedding_dim))

        with open(path) as f:
            for line in f:
                word, vector = line.split()
                if word in self.vectorizer.word_index:
                    idx = self.wtoi(word)
                    self.weights[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]

    def word_to_index(self, word):
        #word to index
        return self.wtoi[word]

    def index_to_word(self, idx):
        #index to word
        return self.itow[idx]

    def train_fasttext(self,
                       path,
                       sg=1,
                       embedding_dim=300,
                       min_count=2,
                       max_vocab_size=30000,
                       seed=42,
                       epochs=10,
                       workers=4,
                       lowercase=False,
                       full=False):

        sentences = self.sentence_train.values

        self.fasttext_model = FastText(sg=sg,
                                       size=embedding_dim,
                                       min_count=min_count,
                                       max_vocab_size=max_vocab_size,
                                       seed=seed,
                                       workers=workers)

        tokenized = list(self._gen_sentences(sentences))

        print('Building vocabulary for fasttext model...')
        self.fasttext_model.build_vocab(sentences=tokenized)

        print('Training fasttext model...')
        self.fasttext_model.train(sentences=tokenized,
                                  total_examples=len(tokenized),
                                  epochs=epochs)
        self.word_vectors = self.fasttext_model.wv

        counts = Counter({
            word: vocab.count
            for (word, vocab) in self.word_vectors.vocab.items()
        })

        self.wtoi = {t[0]: i+1 for i,t in enumerate(counts.most_common(max_vocab_size))}
        self.itow = {v: k for k, v in self.wtoi.items()}
        
        self.tokenized_train = [[self.wtoi.get(word, 0) for word in sentence]
             for sentence in tokenized]

        tok_test = list(self._gen_sentences(self.sentence_test.values))
        self.tokenized_test = [[self.wtoi.get(word, 0) for word in sentence]
             for sentence in tok_test]

        self.pad_and_refresh()

        self.save_fasttext(path)
        self.create_embedding_matrix(embedding_dim)

    def create_embedding_matrix(self, embedding_dim):
        self.weights = np.zeros((self.vocab_size, embedding_dim))

        for word, i in self.wtoi.items():
            if i >= 10000:
                continue
            try:
                embedding_vector = self.word_vectors[word]
                # words not found in embedding index will be all-zeros.
                self.weights[i] = embedding_vector
            except:
                pass

    def save_fasttext(self, path):
        model_path = os.path.join(path, 'fasttext.model')
        self.fasttext_model.save(model_path)

    def _gen_sentences(self, sentences, lowercase=False):
        for s in sentences:
            yield (list(tokenize(s, lowercase=lowercase)))