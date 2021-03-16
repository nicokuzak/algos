from joblib import dump, load

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.backend import clear_session
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras import layers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from modeling.supervised.data_cleaning import Dataset, Tabular, Text


def randomize_x_y(dims=1, size=20):
    X = np.random.rand(size, dims)
    y = np.random.rand(size)
    return X, y


# model = RandomForestClassifier(n_estimators=100)
# model = SVC()
# model = GradientBoostingClassifier()
# model = KNeighborsClassifier(n_neighbors = 3)
# model = GaussianNB()
# model = LogisticRegression()


class SklearnModel:
    def __init__(self, model, ds: Dataset):
        self.model = model
        if isinstance(self.model, str):
            self.load(self.model)
        self.ds = ds

    def train(self):
        return self.model.fit(self.ds.X_train, self.ds.y_train)

    def score(self, X, y):
        return self.model.score(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        dump(self.model, path)

    def load(self, path):
        self.model = load(path)

    def metrics(self, ret='dict'):
        assert ret in ['dict', 'df']
        y_pred = self.predict(self.ds.X_test)
        report = classification_report(self.ds.y_test,
                                       y_pred,
                                       digits=4,
                                       output_dict=True)
        if ret == 'dict':
            return report
        return pd.DataFrame(report).reset_index()

    def cm(self, path):
        y_pred = self.predict(self.ds.X_test)
        cm = confusion_matrix(self.ds.y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)

        cm_plot = disp.plot(cmap=plt.get_cmap('Blues'),
                            ax=None,
                            xticks_rotation=90,
                            values_format='d')

        cm_plot.figure_.savefig(path, bbox_inches='tight')


class NeuralNetwork:
    def __init__(self, ds: Dataset):
        self.ds = ds

    def create_model(self, typ):
        clear_session()
        if typ == 'Sequential':
            self.model = Sequential()

    def add_final_layer(self):
        if self.ds.y_train.nunique() == 2:
            self.model.add(layers.Dense(1, activation='sigmoid'))
            self.model.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])

    def sequential(self, layers, activation='relu'):
        """Create a sequential Keras model"""
        self.create_model('Sequential')
        self.add_dense(layers, activation)
        self.add_final_layer()

    def add_dense(self, filters, activation='relu'):
        for x in filters:
            self.model.add(layers.Dense(x, activation=activation))

    def add_embedding_layer(self):
        if self.ds.weights is not None:
            self.model.add(
                layers.Embedding(input_dim=self.ds.vocab_size,
                                 output_dim=self.ds.weights.shape[1],
                                 input_length=self.ds.X_train.shape[1],
                                 weights=[self.ds.weights]))
        else:
            self.model.add(
                layers.Embedding(input_dim=self.ds.vocab_size,
                                 output_dim=self.ds.weights.shape[1],
                                 input_length=self.ds.X_train.shape[1]))

    def embedding_to_sequential(self, filters, activation='relu'):
        self.create_model('Sequential')
        self.add_embedding_layer()
        self.model.add(layers.GlobalMaxPool1D())
        self.add_dense(filters)
        self.add_final_layer()

    def cnn(self, cnn_filters, kernel_sizes, dense_filters, activation='relu'):
        self.create_model('Sequential')
        self.add_embedding_layer()
        self.add_cnn(cnn_filters, kernel_sizes)
        self.model.add(layers.GlobalMaxPool1D)
        self.add_dense(dense_filters)
        self.add_final_layer()

    def grid_search(self, param_grid, **kwargs):
        #kwargs: epoch=10, verbose=False
        #not sure if this works, might need to return model
        if 'kernel_size' in param_grid:
            self.model = KerasClassifier(build_fn=self.cnn, **kwargs)
        elif 'filters' in param_grid:
            self.model = KerasClassifier(build_fn=self.embedding_to_sequential, **kwargs)
        else:
            self.model = KerasClassifier(build_fn=self.sequential, **kwargs)
        grid = RandomizedSearchCV(estimator=self.model, param_distributions=param_grid, verbose=1)
        grid.fit(self.ds.X_train, self.ds.y_train)
        test_accuracy = grid.score(self.ds.X_test, self.ds.y_test)
        print(f'Test accuracy for best model is {test_accuracy}')
        return grid


    def add_cnn(self, filters, kernels):
        assert len(filters) == len(kernels)
        for i in range(len(filters)):
            self.model.add(layers.Conv1D(filters[i], kernels[i]))

    def train(self, **kwargs):
        self.model.fit(self.ds.X_train, self.ds.y_train, **kwargs)

    def evaluate(self):
        trainl, traina = self.model.evaluate(self.ds.X_train,
                                             self.ds.y_train,
                                             verbose=False)
        vall, vala = self.model.evaluate(self.ds.X_test,
                                         self.ds.y_test,
                                         verbose=False)
        print(f'Training loss {trainl}, training accuracy {traina}')
        print(f'Validation loss {vall}, validation accuracy {vala}')
        return vall, vala

    def plot_history(self, path):
        metrics = ['acc', 'val_acc', 'loss', 'val_loss']
        metric_dict = {
            'acc': ['Training Accuracy', 'b'],
            'val_acc': ['Validation Accuracy', 'r'],
            'loss': ['Training Loss', 'b'],
            'val_loss': ['Validation Loss', 'r']
        }

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title('Training and validation accuracy')
        plt.legend()
        for metric in metrics:
            if 'loss' in metric:
                plt.subplot(1, 2, 2)
            met = self.model.history.history[metric]
            x = range(1, len(met) + 1)
            plt.plot(x, met, metric_dict[met][1], label=metric_dict[met][0])

        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(path)