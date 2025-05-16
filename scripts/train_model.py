import sys
import os

# Adiciona a raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from model.clustering import save_model, train_kmeans
from model.utils import assign_clusters, add_pca_components
from model.pipeline import preprocess_data

MODEL_PATH = 'models/kmeans_model.pkl'
DATA_PATH = r"C:\Users\usr\OneDrive - instituto de patologia cirúrgica e molecular\BASES\BASE_INDICADORES.csv"
OUTPUT_PATH = 'data/clientes_com_clusters.csv'

def main():
    # 1. Carregar dados brutos
    df = pd.read_csv(DATA_PATH, sep=';')

    # 2. Pré-processamento (separado no pipeline)
    df_processed = preprocess_data(df)  # se existir; senão pule

    # 3. Treinar modelo
    model = train_kmeans(df_processed)

    # 4. Salvar modelo treinado
    save_model(model, MODEL_PATH)

    # 5. Aplicar modelo e gerar colunas para visualização
    df_cluster = assign_clusters(df_processed, model)
    df_cluster = add_pca_components(df_cluster)

    # 6. Salvar base com clusters e PCA
    df_cluster.to_csv(OUTPUT_PATH, index=False)
    print(f"Modelo salvo em {MODEL_PATH}")
    print(f"Dados com clusters salvos em {OUTPUT_PATH}")

if __name__ == '__main__':
    main()