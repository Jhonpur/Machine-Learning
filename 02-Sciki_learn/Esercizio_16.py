import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

'''ESERCIZIO 16
Crea un modello di regressione polinomiale su un dataset di regressione e sperimenta l'effetto di vari gradi di polinomio
sul fenomeno dell'overfitting e dell'underfitting. Visualizza le curve di apprendimento e confronta i risultati.'''
def load_data():
    data = fetch_california_housing()
    X = data.data[:, [0]]  # usiamo solo la prima feature per semplicità
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=0)

def polynomial_regression(X_train, X_test, y_train, y_test, grado):
    model = make_pipeline(PolynomialFeatures(grado), StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    return train_error, test_error

def curva_apprendimento(gradi, X_train, X_test, y_train, y_test):
    train_errors = []
    test_errors = []
    for grado in gradi:
        tr_err, ts_err = polynomial_regression(X_train, X_test, y_train, y_test, grado)
        train_errors.append(tr_err)
        test_errors.append(ts_err)
    plt.plot(gradi, train_errors, label="Train MSE")
    plt.plot(gradi, test_errors, label="Test MSE")
    plt.xlabel("Grado polinomiale")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    gradi = range(1, 10)
    curva_apprendimento(gradi, X_train, X_test, y_train, y_test)
