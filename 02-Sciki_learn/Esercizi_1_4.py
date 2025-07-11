import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  accuracy_score,confusion_matrix

'''ESERCIZIO 1
Carica il dataset Iris incluso in scikit-learn e visualizza le prime 5 righe dei dati.'''
# Carica il dataset Iris
iris = load_iris()

# Creare un DataFrame per visualizzare i dati in modo chiaro
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Aggiungiamo la colonna della specie (target)
df['species'] = iris.target

# Stampiamo solo le prime 5 righe del dataset
print(df.head(5))

'''ESERCIZIO 2
Dividi il dataset Iris in un set di addestramento (80%) e un set di test (20%).'''
# Dividiamo i dati in X (caratteristiche) e y (target/specie)
X = iris.data
y = iris.target

# Dividiamo in 80% per l'addestramento e 20% per il test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''ESERCIZIO 3
Crea un modello di classificazione utilizzando scikit-learn,
ad esempio un classificatore SVM, e addestralo con il set di addestramento.'''
# Creiamo il modello k-NN con k=3
model = KNeighborsClassifier(n_neighbors=3)

# Addestriamo il modello sui dati di addestramento
model.fit(X_train, y_train)

# Facciamo previsioni sui dati di test
y_pred = model.predict(X_test)

# Mostriamo alcune previsioni
print("Previsioni per i dati di test:", y_pred)
print("Valori reali delle specie:", y_test)

'''ESERCIZIO 4
Valuta il modello di classificazione creato nell’esercizio 3
utilizzando il set di test e calcola l'accuratezza.'''
# Calcoliamo la precisione del modello
accuracy = accuracy_score(y_test, y_pred)  # La precisione indica la percentuale di previsioni corrette
print(f"Precisione del modello: {accuracy * 100:.2f}%")

# Mostriamo la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)  # La matrice di confusione mostra il confronto tra le previsioni corrette e quelle sbagliate.
print("Matrice di confusione:")
print(conf_matrix)