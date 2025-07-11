from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score

'''ESERCIZIO 18
Carica un dataset adatto al clustering (come Iris) e sperimenta diversi algoritmi di clustering,
come il clustering K-Means, il clustering gerarchico e il DBSCAN. Valuta e confronta i risultati.'''
def load_data():
    data = load_iris()
    return data.data, data.target

def clustering_kmeans(X, y):
    pred = KMeans(n_clusters=3, random_state=0).fit_predict(X)
    return adjusted_rand_score(y, pred)

def clustering_gerarchico(X, y):
    pred = AgglomerativeClustering(n_clusters=3).fit_predict(X)
    return adjusted_rand_score(y, pred)

def clustering_dbscan(X, y):
    pred = DBSCAN().fit_predict(X)
    return adjusted_rand_score(y, pred)

if __name__ == '__main__':
    X, y = load_data()
    print(f"KMeans ARI: {clustering_kmeans(X, y):.4f}")
    print(f"Gerarchico ARI: {clustering_gerarchico(X, y):.4f}")
    print(f"DBSCAN ARI: {clustering_dbscan(X, y):.4f}")
