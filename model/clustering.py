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


def save_model(model, path="models/cluster_model.pkl"):
    dump(model, path)

    
def load_model(path="models/cluster_model.pkl"):
    return load(path)
