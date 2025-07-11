from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''ESERCIZIO 19
Sperimenta la differenza tra la regressione logistica e un modello SVM per un problema di classificazione binaria.
Valuta le prestazioni di entrambi gli algoritmi su un dataset e confrontali.'''
def load_data():
    X, y = load_iris(return_X_y=True)
    mask = y < 2  # classi 0 e 1
    return X[mask], y[mask]

def logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

def svm_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = SVC()
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

if __name__ == '__main__':
    X, y = load_data()
    acc_log = logistic_regression(X, y)
    acc_svm = svm_classification(X, y)
    print(f"Logistic Regression accuracy: {acc_log:.4f}")
    print(f"SVM accuracy: {acc_svm:.4f}")
