from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

'''ESERCIZIO 15
Utilizza la K-Fold Cross-Validation per valutare le prestazioni di un algoritmo di classificazione o regressione su un dataset.
Calcola la media delle misure di prestazione (ad esempio, accuratezza per la classificazione o errore quadratico medio per la regressione)
su diverse ripartizioni del dataset.'''
def load_data():
    iris = load_iris()
    return iris.data, iris.target

def kfold_valutazione(X, y):
    modello = LogisticRegression(max_iter=200)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(modello, X, y, cv=kf, scoring='accuracy')
    return np.mean(scores)

if __name__ == '__main__':
    X, y = load_data()
    media_accuracy = kfold_valutazione(X, y)
    print(f"Media accuratezza con K-Fold: {media_accuracy:.4f}")
