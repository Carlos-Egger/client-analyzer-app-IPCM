import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configuração inicial
st.set_page_config(page_title="Análise de Clusters de Clientes", layout="wide")
# CSS para impedir digitação e alterar cursor
st.markdown("""
    <style>
    .stSelectbox > div > div > input {
        pointer-events: none;
        caret-color: transparent;
    }
    .stSelectbox > div > div {
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# --- Caminho dos dados ---
DATA_PATH = "data/clientes_com_clusters.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# --- Funções das seções ---
def visualizacao_clusters(df):
    st.title("📍 Visualização dos Clusters")

    # Filtro interativo por cluster
    clusters = sorted(df['cluster'].unique())
    cluster_selecionado = st.multiselect("Selecione o(s) cluster(s):", clusters, default=clusters)

    df_filtrado = df[df['cluster'].isin(cluster_selecionado)]

    # Gráfico de dispersão
    st.subheader("Gráfico de Dispersão (PCA)")

    cluster_representa = {
    0: "Fraqueza",
    1: "Ameaça",
    2: "Força",
    3: "Oportunidades"
}

    fig, ax = plt.subplots(figsize=(10, 5))

    for c in sorted(df_filtrado['cluster'].unique()):
        cluster_df = df_filtrado[df_filtrado['cluster'] == c]
        nome_cluster = cluster_representa.get(c, f"Cluster {c}")  # fallback caso c não esteja no dicionário
        ax.scatter(cluster_df['pca1'], cluster_df['pca2'], label=f"{nome_cluster}", alpha=0.6)

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Clusters dos Clientes")
    ax.legend()
    st.pyplot(fig)

    # Tabela dos dados
    st.subheader("📄 Dados Filtrados")
    st.dataframe(df_filtrado.head(20))


def analise_descritiva(df):
    st.title("📊 Análise Descritiva dos Clusters")

    st.subheader("Média dos Atributos por Cluster")
    st.dataframe(df.groupby("cluster").mean(numeric_only=True).round(2))


def sobre():
    st.title("ℹ️ Sobre o App")
    st.markdown("""
    Este aplicativo foi desenvolvido para realizar uma **Análise Detalhada sobre os clientes do Laboratório** com base nos dados disponibilizados pelo aplis.

    **TECNOLOGIAS UTILIZADAS:**
    - **Streamlit** para a interface
    - **KMeans** para agrupamento
    - **PCA** para redução de dimensionalidade

    Criado por: Carlos Egger -  Cientista de Dados Lab IPCM
    """)

# --- Sidebar (menu) ---
st.sidebar.title("🔍 Navegação")
pagina = st.sidebar.selectbox(
    "Escolha uma página:",
    ["Visualização de Clusters", "Análise Descritiva", "Sobre o App"],
    index=0
)

# --- Controle da navegação ---
if pagina == "Visualização de Clusters":
    visualizacao_clusters(df)
elif pagina == "Análise Descritiva":
    analise_descritiva(df)
elif pagina == "Sobre o App":
    sobre()
