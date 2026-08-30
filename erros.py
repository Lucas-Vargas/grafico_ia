import pandas as pd

# Carregamento da base
df = pd.read_csv("spam.csv", encoding="latin-1")

print("Dimensões da base:", df.shape)
print("Colunas:", df.columns.tolist())

duplicatas_exatas = df[df.duplicated(keep=False)]

print("Quantidade de linhas duplicadas:", df.duplicated().sum())

duplicatas_por_chave = df[
    df.duplicated(subset=["v1", "v2"], keep=False)
]

print(
    "Duplicatas por rótulo e mensagem:",
    df.duplicated(subset=["v1", "v2"]).sum()
)

print(duplicatas_por_chave)

colunas_extras = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]

for coluna in colunas_extras:
    total_ausentes = df[coluna].isna().sum()
    total_preenchidos = df[coluna].notna().sum()
    proporcao_ausentes = df[coluna].isna().mean() * 100

    print(f"\nColuna: {coluna}")
    print(f"Valores preenchidos: {total_preenchidos}")
    print(f"Valores ausentes: {total_ausentes}")
    print(f"Proporção de ausentes: {proporcao_ausentes:.2f}%")