import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
#colunas existentes:
#Ano, Semestre, Cód. Disciplina, Cód. Turma, Situação, %, Alunos, Professor, Cód. Curso, Curso

# 1. Carregar os dados
caminho_pasta = 'datasets'
dataframes = []

for arquivo in os.listdir(caminho_pasta):
    if arquivo.endswith('.xlsx'):
        caminho_completo = os.path.join(caminho_pasta, arquivo)
        df = pd.read_excel(caminho_completo)
        dataframes.append(df)

df_merged = pd.concat(dataframes, ignore_index=True)

# 2. Pré-processar os dados
# Remover colunas irrelevantes
df_merged = df_merged.drop(['Semestre', 'Cód. Curso', 'Cód. Turma'], axis=1)
# Remover duplicatas
df_merged = df_merged.drop_duplicates()
# Lidar com valores ausentes
df_merged = df_merged.fillna(0)
######### fim do pre preocessamento
df_origin = df_merged.copy()

#relacoes graficas
#Criar um mapeamento de Professor para ID
professor_to_id = {professor: idx for idx, professor in enumerate(df_merged['Professor'].unique())}
df_merged['Professor_ID'] = df_merged['Professor'].map(professor_to_id)

# correlação de reprovacao por professor
df = df_merged.groupby(['Professor_ID', 'Situação'])['Alunos'].sum().unstack(fill_value=0)
df['Reprovado'] = df.drop(columns=['Aprovado']).sum(axis=1)

#professor_id
def gerar_professor_id(index):
    return index

# Adicionando a coluna professor_id com valores gerados pela função
df['professor_id'] = df.index.to_series().apply(gerar_professor_id)
df = df.filter(items=['Aprovado', 'Reprovado', 'professor_id'])
df['%'] = df['Aprovado'] / (df['Aprovado'] + df['Reprovado']) * 100

plt.figure(figsize=(30,15))
plt.plot(df['professor_id'], df['%'], marker='o', linestyle='', color='b')
plt.title("Aprovacao por Professor ID")
plt.xlabel("Professor ID")
plt.ylabel("%")
plt.xticks(df['professor_id'])
plt.grid(True)
plt.savefig('proprocao_aprovacao_por_professor.pdf', format='pdf')

df = df_origin

df_aprovado = df[df['Situação'] == 'Aprovado']

# Calcular a média da coluna "%" por Ano
df = df_aprovado.groupby('Ano')['%'].mean().reset_index()
print(df)
X = df['Ano'].values.reshape(-1, 1) 
y = df['%'].values


model = LinearRegression()
model.fit(X, y)


anos_futuros = np.array([2020, 2024, 2025]).reshape(-1, 1)
previsoes = model.predict(anos_futuros)

# Exibir as previsões
for ano, previsao in zip(anos_futuros.flatten(), previsoes):
    print(f"Previsão de {ano}: {previsao:.2f}%")


#grafico de Situacao por professor abosuluto
professor_to_id = {professor: idx for idx, professor in enumerate(df_merged['Professor'].unique())}

df_merged['Professor_ID'] = df_merged['Professor'].map(professor_to_id)


df_grouped = df_merged.groupby(['Professor_ID', 'Situação'])['Alunos'].sum().unstack(fill_value=0)


ax = df_grouped.plot(kind='bar', stacked=True, figsize=(30, 15))
ax.set_title('Número de Alunos Aprovados e Reprovados por Professor (ID)')
ax.set_xlabel('ID do Professor')
ax.set_ylabel('Número de Alunos')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


df_proporcao = df_grouped.div(df_grouped.sum(axis=1), axis=0)

#grafico de Situacao por professor relativo
ax = df_proporcao.plot(kind='bar', stacked=True, figsize=(10, 6))
ax.set_title('Proporção de Aprovados e Reprovados por Professor (ID)')
ax.set_xlabel('ID do Professor')
ax.set_ylabel('Proporção de Alunos')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()