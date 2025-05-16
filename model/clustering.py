import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns


def train_kmeans(df, n_clusters=4, features=None, random_state=42):
    if features is None:
        features = df.select_dtypes(include='number').columns.tolist()

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(X_scaled)

    score = silhouette_score(X_scaled, model.labels_)
    print(f"Silhouette Score: {score:.4f}")

    return model


def train_dbscan(df, eps=0.5, min_samples=5, features=None, scale=True):
    """
    Treina o DBSCAN e retorna o modelo e o score de silhouette.
    """
    if features is None:
        features = df.select_dtypes(include='number').columns.tolist()

    X = df[features].drop('ticket_medio_laminas')

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    if len(set(labels)) > 1 and len(set(labels)) < len(X):
        score = silhouette_score(X, labels)
    else:
        score = np.nan  # Silhouette score não aplicável

    print(f"Clusters encontrados: {n_clusters}")
    print(f"Ruídos detectados: {n_noise}")
    print(f"Silhouette Score: {score:.4f}" if not np.isnan(score) else "Silhouette Score: Não aplicável")

    return model, labels, score


def save_model(model, path="models/cluster_model.pkl"):
    dump(model, path)

    
def load_model(path="models/cluster_model.pkl"):
    return load(path)


def scatter_plot(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='Set2')
    plt.title('Visualização dos Clusters (PCA)')
    plt.show()