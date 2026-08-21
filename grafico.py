import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlopen
import zipfile
import io

# --------------------------------------------------
# 1. Baixar o dataset
# --------------------------------------------------

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

response = urlopen(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.read()))

# Arquivo original dentro do ZIP
with zip_file.open("SMSSpamCollection") as file:
    df = pd.read_csv(
        file,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

# --------------------------------------------------
# 2. Visualizar os primeiros dados
# --------------------------------------------------

print("Primeiras mensagens:")
print(df.head())

print("\nNúmero de registros:", len(df))

# --------------------------------------------------
# 3. Frequência de cada classe
# --------------------------------------------------

frequencia = df["label"].value_counts()

print("\nQuantidade por classe:")
print(frequencia)

# --------------------------------------------------
# 4. Estatísticas sobre o tamanho das mensagens
# --------------------------------------------------

df["message_length"] = df["message"].str.len()

print("\nEstatísticas do tamanho das mensagens:")
print(df["message_length"].describe())

# --------------------------------------------------
# 5. Gráfico
# --------------------------------------------------

plt.figure(figsize=(7, 5))

frequencia.plot(
    kind="bar",
    color=["steelblue", "tomato"]
)

plt.title("Distribuição de mensagens SMS")
plt.xlabel("Tipo de mensagem")
plt.ylabel("Quantidade de mensagens")

plt.xticks(
    ticks=[0, 1],
    labels=["Ham (legítima)", "Spam"],
    rotation=0
)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()