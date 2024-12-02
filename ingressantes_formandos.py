import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

caminho_pasta = 'datasets\Ingressantes e Formandos'
dataframes = []

for arquivo in os.listdir(caminho_pasta):
    if arquivo.endswith('.xls'):
        caminho_completo = os.path.join(caminho_pasta, arquivo)
        df = pd.read_excel(caminho_completo)
        dataframes.append(df)

df_merged = pd.concat(dataframes, ignore_index=True)
