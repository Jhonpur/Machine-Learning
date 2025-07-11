import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score

'''ESERCIZIO 11
Carica un dataset di classificazione (come Iris o Breast Cancer) e
sperimenta diversi algoritmi di classificazione disponibili in scikit-learn,
come Support Vector Machine (SVM), Random Forest, e k-Nearest Neighbors (k-NN).
Valuta e confronta le prestazioni di ciascun algoritmo.'''
def load_iris_data():
    print("Sto caricando il dataset Iris...")
    iris = load_iris()
    x = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="Species")
    print(x.head())
    return x, y

def train_models(X_train, X_test, y_train, y_test):
    print('Sto confrontando i modelli...')

    models = {
        "SVM (linear)": SVC(kernel='linear'),
        "SVM (rbf)": SVC(kernel='rbf'),
        "SVM (poly)": SVC(kernel='poly', degree=3),
        "Random Forest": RandomForestClassifier(),
        "k-NN": KNeighborsClassifier()
    }
    results = {}

    print('Li sto valutando...')
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name}: Accuracy = {acc:.4f}")
    return results

'''ESERCIZIO 13
Prendi uno dei dataset utilizzati negli esercizi 11 o 12 e utilizza la ricerca degli iperparametri per trovare i migliori parametri
per uno degli algoritmi di machine learning. Ad esempio,
puoi utilizzare GridSearchCV per ottimizzare gli iperparametri di un modello SVM o di una regressione Ridge.'''
def svm_gridsearch(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
			'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
			'kernel': ['linear','rbf','poly']}
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    grid.fit(X_train, y_train)
    print("Migliori parametri trovati:")
    print(grid.best_params_)
    print(f"Accuratezza migliore nel cross-validation: {grid.best_score_:.4f}")
    return grid.best_estimator_

if __name__ == '__main__':
    # 1) Carica i dati
    x, y = load_iris_data()

    # 2) Suddividi in train/test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # 3) Addestra e valuta i modelli
    results = train_models(X_train, X_test, y_train, y_test)
    
    # GridSearchCV per SVM
    print("\nOttimizzazione SVM con GridSearchCV...")
    best_svm = svm_gridsearch(X_train, y_train)

    # Valuta SVM ottimizzato
    y_pred = best_svm.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred)
    print(f"Accuratezza del miglior modello SVM su test set: {final_acc:.4f}")