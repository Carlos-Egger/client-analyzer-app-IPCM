import pandas as pd

encoder_exame_tipo = {
    'ANATOMO PATOLOGICO': 4,
    'CITOPATOLOGIA' : 3,
    'IMUNO-HISTOQUIMICA': 5,
    'REVISAO INTERNA': 1,
    'BIOLOGIA MOLECULAR interno': 2,
    'IMUNO-CITOQUIMICA': 5,
    'BIOLOGIA MOLECULAR externo': 2
}

encoder_exame = {
    "BIOPSIA SIMPLES": 0.85,
    "BIOPSIA GASTRICA": 0.85,
    'PELE': 0.8,
    'CITOLOGIA DE LIQUIDOS': 1,
    'IMUNO-HISTOQUIMICA INTERNA GERAL': 0.9,
    "PECA CIRURGICA COMPLEXA": 1,
    "PECA CIRURGICA SIMPLES": 0.9,
    'BIOPSIA DE CORNEA': 0.85,
    'IMUNO-HISTOQUIMICA INTERNA MAMA': 1,
    'REVISAO DE LAMINA EXTERNO (BLOCO + LAMINA)': 1,
    'IMUNO-HISTOQUIMICA INTERNA LINFOMA': 0.8,
    'IMUNO-HISTOQUIMICA EXTERNA GERAL (BLOCO+LAMINA)': 0.9,
    'CITOLOGIA VAGINAL CONVENCIONAL': 1,
    'PD-L 1 (22C3) INTERNO': 1,
    'IMUNO-HISTOQUIMICA EXTERNA  GERAL (BLOCO)': 0.9,
    'IMUNO-HISTOQUIMICA EXTERNA  MAMA (BLOCO)': 1,
    'BIOPSIA DE MEDULA OSSEA': 0.9,
    'REVISÃO DE LÂMINA EXTERNO (BLOCO)': 1,
    'SISH - Her2 INTERNO': 1,
    'REVISAO DE LAMINA EXTERNA DE IMUNO-HISTOQUIMICA': 1,
    'IMUNO-HISTOQUIMICA EXTERNA LINFOMA (BLOCO)': 0.8,
    'PECA CIRURGICA COMPLEXA - MEMBROS': 1,
    'IMUNO-HISTOQUIMICA EXTERNA MAMA (BLOCO+LAMINA)': 1,
    'URINA 3 AMOSTRAS': 1,
    'BIOPSIA HEPATICA TX': 0.85,
    'REVISAO DE LAMINA INTERNA': 1,
    'BIÓPSIA RENAL TX': 0.85,
    'CITOLOGIA - LAMINAS': 1,
    'REVISÃO DE LÂMINA INTERNA DE IMUNO-HISTOQUÍMICA' : 0.7,
    'COLORAÇÃO ESPECIAL AVULSA': 1,
    'IMUNO-CITOQUÍMICA INTERNA': 1,
    'PECA CIRURGICA COMPLEXA - SVO': 1,
    'PECA CIRURGICA COMPLEXA - IML': 1,
    'IMUNO-HISTOQUIMICA EXTERNA LINFOMA (BLOCO+LAMINA)': 0.8,
    'CITOLOGIA DE LIQUIDOS/LAMINAS': 1,
    'SISH - Her2 EXTERNO': 1,
    'PD-L 1 (22C3) EXTERNO': 1,
    'MAMA': 1,
    'BIÓPSIA DE MAMA': 1,
    'BIOPSIA DE MAMA': 1,
    'MAMOTOMIA': 1,
    }


def drop_irrelevant_coluns_and_rows(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop=['NomPatologista', 'NomMedico', 'Ano', 'Tat', 'VlrBruto', "VlrRecebido", "DtaRecebido", "NomLocalOrigem", "DtaSolicitacao", "DtaFinalizacao"]
    df = df.drop(columns=columns_to_drop)
    conv_to_drop = ['CORTESIA', 'FUSEX', 'ONG ZOE', 'UNACON TUCURUI', 'PARTICULAR EXTERNO']
    df = df[~df['NomFontePagadora'].isin(conv_to_drop)]
    return df


def treat_outliers_nan(df: pd.DataFrame) -> pd.DataFrame:
    df['QtdLam'].dropna(inplace=True)
    df['VlrLiquido'].dropna(inplace=True)
    df = df[df['QtdLam'] > 0]
    df = df[df['QtdLam'] < 60]
    df = df[df['VlrLiquido'] > 0]
    return df


def treat_exames(df: pd.DataFrame) -> pd.DataFrame:
    exames = ["BIOPSIA SIMPLES","BIOPSIA GASTRICA", "PELE", "CITOPATOLOGIA", 'IMUNO-HISTOQUIMICA', 'PECA CIRURGICA COMPLEXA', 'PECA CIRURGICA SIMPLES']
    df = df[df['NomExame'] != "LAMINA AVULSA"]
    df['ExameSeparado'] = df.apply(lambda row: row['NomExame'] if row['NomExameTipo'] == 'ANATOMO PATOLOGICO' else row['NomExameTipo'], axis=1)
    df = df[df['ExameSeparado'].isin(exames)]
    dummies = pd.get_dummies(df['ExameSeparado']).astype(int)
    df = pd.concat([df, dummies], axis=1)
    return df


def gen_calculate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['Lam_Price'] = df['VlrLiquido'] / df['QtdLam']
    df['PesoExameTipo'] = df['NomExameTipo'].map(encoder_exame_tipo)
    df['PesoExame'] = df['NomExame'].map(encoder_exame)
    df['PesoFim'] = df['PesoExame'] * df['PesoExameTipo']
    df['VlrPonderado'] = df['VlrLiquido'] * df['PesoFim']
    df['LamPricePonderado'] = df['Lam_Price'] * df['PesoFim']
    df['Mes'] = pd.to_datetime(df['Mes'])
    df['trimestre'] = df['Mes'].dt.year.astype(str) + '-T' + df['Mes'].dt.quarter.astype(str)
    return df


def aggregate_by_period(df: pd.DataFrame) -> pd.DataFrame:

    df_agg = df.groupby(['Mes', 'NomFontePagadora']).agg(
    VlrLiquidoTotal=('VlrLiquido', 'sum'),
    VlrPonderadoTotal=('VlrPonderado', 'sum'),
    laminas=('QtdLam', 'sum'),
    ticket_medio=('VlrLiquido', 'mean'),
    ticket_medio_laminas=('Lam_Price', 'mean'),
    ticket_medio_laminas_ponderado=('LamPricePonderado', 'mean'),
    QtdReq=('VlrLiquido', 'count'),  # ou usar qualquer outra coluna da venda
    BiopsiaGastrica=('BIOPSIA GASTRICA', 'sum'),
    BiopsiaSimples=('BIOPSIA SIMPLES', 'sum'),
    Citologia=('CITOPATOLOGIA', 'sum'),
    Imuno=('IMUNO-HISTOQUIMICA', 'sum'),
    PecaComplexa=('PECA CIRURGICA COMPLEXA', 'sum'),
    PecaSimples=('PECA CIRURGICA SIMPLES', 'sum'),
    Pele=('PELE', 'sum'),
).reset_index()
    
    df_agg['anatomo'] = df_agg[['BiopsiaGastrica', 'BiopsiaSimples', 'PecaComplexa', 'PecaSimples', 'Pele']].sum(axis=1)
    df_agg['ConvercaoImuno'] = (df_agg['Imuno']/df_agg['anatomo'])
    df_agg = df_agg.drop(columns='anatomo')

    return df_agg


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_irrelevant_coluns_and_rows(df)
    df = treat_outliers_nan(df)
    df = gen_calculate_columns(df)
    df = treat_exames(df)
    #df = aggregate_by_period(df)

    return df


df = pd.read_csv(r"C:\Users\usr\OneDrive - instituto de patologia cirúrgica e molecular\BASES\BASE_INDICADORES.csv", sep=';')

df = preprocess_data(df)

df.to_csv('testar.csv', index=False)