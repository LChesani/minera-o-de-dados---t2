import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

caminho_pasta = 'datasets\Ingressantes e Formandos'
dataframes = []

for arquivo in os.listdir(caminho_pasta):
    if arquivo.endswith('.xls'):
        caminho_completo = os.path.join(caminho_pasta, arquivo)
        df = pd.read_excel(caminho_completo)
        dataframes.append(df)

df_origin = pd.concat(dataframes, ignore_index=True)
df = df_origin.copy()

df = df[df['ANO'] != 'TOTAL']
df = df.drop(columns=['NOME_UNIDADE', 'NIVEL_CURSO', 'FORMADOS'])

df = pd.get_dummies(df, columns=['COD_CURSO', 'SEXO'], drop_first=True)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)


#uso do k-means


kmeans = KMeans(n_clusters=3, random_state=7)

kmeans.fit(df_scaled)

df['Cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_

df = df.groupby('ANO').mean().reset_index()
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['ANO'], df['INGRESSANTES'],  c=df['Cluster'], cmap='viridis', s=100, alpha=0.7, edgecolor='k', label='Pontos')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centróides')
plt.title('Clusters Formados pelo K-means', fontsize=16)
plt.ylabel('INGRESSANTES', fontsize=12)
plt.xlabel('ANO', fontsize=12)
plt.colorbar(scatter, label='Cluster baseado no Sexo')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('k_means_ingressantes_femininos_por_ano.pdf', format='pdf')
plt.show()

df = df_origin.copy()

total_por_ano = df.groupby('ANO')['INGRESSANTES'].transform('sum')
df['PROPORCAO'] = df['INGRESSANTES'] / total_por_ano * 100
df_proporcao = df.pivot_table(
    index='ANO', 
    columns='SEXO', 
    values='PROPORCAO'
).reset_index()

df_proporcao.columns = ['ANO', 'PROPORCAO_F', 'PROPORCAO_M']

df = df_proporcao.copy()
df = df[df['ANO'] != 'TOTAL']

X = df[['ANO']]  # Variável independente (ano)
y = df['PROPORCAO_F']  # Variável dependente (proporção feminina)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

prev = pd.DataFrame([2025, 2030, 2040], columns=['ANO'])
previsoes = model.predict(prev)
print(previsoes)