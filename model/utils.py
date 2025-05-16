from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def assign_clusters(df: pd.DataFrame, model, features=None, cluster_col='cluster'):
    
    if features is None:
        features = df.select_dtypes(include='number').columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    labels = model.fit_predict(X_scaled)
    df[cluster_col] = labels

    return df


def add_pca_components(df: pd.DataFrame, features=None, n_components=2):

    if features is None:
        features = df.select_dtypes(include='number').columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    for i in range(n_components):
        df[f'pca{i+1}'] = X_pca[:, i]

    return df
