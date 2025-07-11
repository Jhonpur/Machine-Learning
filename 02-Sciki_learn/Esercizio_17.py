import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

'''ESERCIZIO 17
Prendi un dataset e applica tecniche di feature engineering, come la creazione di nuove feature o
la trasformazione delle feature esistenti. Valuta come queste modifiche influenzano le prestazioni del tuo modello di machine learning.'''
def load_data():
    data = fetch_california_housing()
    X = data.data
    y = data.target
    return X, y

def feature_engineering(X):
    # aggiungiamo la feature quadratica della prima colonna
    X_new = np.hstack([X, X[:, [0]] ** 2])
    return X_new

def evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

if __name__ == '__main__':
    X, y = load_data()
    mse_originale = evaluate_model(X, y)
    X_eng = feature_engineering(X)
    mse_nuovo = evaluate_model(X_eng, y)

    print(f"MSE originale: {mse_originale:.4f}")
    print(f"MSE con feature engineering: {mse_nuovo:.4f}")
