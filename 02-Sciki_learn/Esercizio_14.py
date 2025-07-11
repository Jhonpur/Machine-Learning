import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

'''ESERCIZIO 14
Sperimenta la differenza tra problemi di classificazione binaria e multiclasse utilizzando un dataset appropriato.
Addestra un classificatore binario e un classificatore multiclasse sullo stesso dataset e confronta i risultati.'''
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return X, y

def classificazione_binaria(X, y):
    # Considera solo due classi: 0 e 1
    binary_filter = y < 2
    X_bin = X[binary_filter]
    y_bin = y[binary_filter]
    X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=0)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def classificazione_multiclasse(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

if __name__ == '__main__':
    X, y = load_data()
    acc_binaria = classificazione_binaria(X, y)
    acc_multiclasse = classificazione_multiclasse(X, y)

    print(f"Accuratezza classificazione binaria (LogisticRegression): {acc_binaria:.4f}")
    print(f"Accuratezza classificazione multiclasse (MultinomialNB): {acc_multiclasse:.4f}")
