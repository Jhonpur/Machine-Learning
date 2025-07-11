from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

'''ESERCIZIO 5
Carica il dataset Breast Cancer incluso in scikit-learn e visualizza le prime 5 righe dei dati.'''
def load_data():
    print('Caricamento del breast Cancer dataset in corso...')
    cancer = load_breast_cancer()

    x = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name='target')
    print(x.head(5))
    return x, y

'''ESERCIZIO 6
Dividi il dataset Breast Cancer in un set di addestramento (70%) e un set di test (30%).'''
def train_test_func(x, y):
    print('Suddivisione dati in addestramento (70%) e test (30%) in corso...')
    return train_test_split(x, y, test_size=0.3, random_state=42)

'''ESERCIZIO 7
Crea un modello di regressione utilizzando scikit-learn, ad esempio un regressore lineare, e addestralo con il set di addestramento.'''
def regression_model(X_train, y_train):
    print('Addestramento del modello in corso...')
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

'''ESERCIZIO 8
Valuta il modello di regressione creato nel punto 7 utilizzando il set di test e calcola l'errore quadratico medio (Mean Squared Error).'''
def metrica_mse(model, X_test, y_test):
    print('Calcolo dell MSE in corso...')
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse:.4f}')
    return mse

if __name__ == '__main__':
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_func(x, y)
    model = regression_model(x_train, y_train)
    metrica_mse(model, x_test, y_test)