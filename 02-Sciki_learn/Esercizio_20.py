from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

'''ESERCIZIO 20
Utilizza algoritmi di ensemble (di classificazione), come Random Forest e Gradient Boosting per migliorare
le prestazioni del tuo modello di machine learning. Addestra modelli singoli e confrontali con il modello ensemble.'''
def load_data():
    return train_test_split(*load_iris(return_X_y=True), test_size=0.2, random_state=0)

def decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

def gradient_boosting(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    print(f"Decision Tree accuracy: {decision_tree(X_train, X_test, y_train, y_test):.4f}")
    print(f"Random Forest accuracy: {random_forest(X_train, X_test, y_train, y_test):.4f}")
    print(f"Gradient Boosting accuracy: {gradient_boosting(X_train, X_test, y_train, y_test):.4f}")
