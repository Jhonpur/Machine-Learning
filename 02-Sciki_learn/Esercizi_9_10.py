from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import os
'''ESERCIZIO 9
Carica un dataset personalizzato in formato CSV utilizzando Pandas e crea un modello di classificazione utilizzando scikit-learn.'''
def lettura_csv():
    print('Lettura dei dati in corso...')
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'clienti_acquisti.csv'))
    print(df.head(5))
    return df

def preprocessing_data(df):
    X = df[['Eta','Reddito_Annuale','Acquisti_Precedenti']]
    y = df['Acquisto']
    return X, y

def train_test_func(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)

def regression_model(X_train, y_train):
    print('Addestramento del modello in corso...')
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

'''ESERCIZIO 9
Valuta il modello di classificazione creato nell’esercizio 9 utilizzando il dataset personalizzato e calcola l'accuratezza.'''
def accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuratezza del modello: {accuracy:.2f}")
    return accuracy

if __name__ == '__main__':
    df = lettura_csv()
    x, y = preprocessing_data(df)
    x_train, x_test, y_train, y_test = train_test_func(x, y)
    model = regression_model(x_train, y_train)
    accuracy(model, x_test, y_test)