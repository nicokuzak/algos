import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

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

class Model:
    def __init__(self, model):
        self.model = model
        if isinstance(self.model, str):
            self.load(self.model)

    def train(self, X, y):
        return self.model.fit(X, y)

    def score(self, X, y):
        return self.model.score(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        dump(self.model, path)
    
    def load(self, path):
        self.model = load(path)
    
    def metrics(self, X, y, ret='dict'):
        assert ret in ['dict', 'df']
        y_pred = self.predict(X)
        report = classification_report(y, y_pred, digits=4, output_dict=True)
        if ret == 'dict':
            return report
        return pd.DataFrame(report).reset_index()
    
    def cm(self, X, y, path):
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm)

        cm_plot = disp.plot(cmap=plt.get_cmap('Blues'),
                            ax=None,
                            xticks_rotation=90,
                            values_format='d')

        cm_plot.figure_.savefig(path, bbox_inches='tight')