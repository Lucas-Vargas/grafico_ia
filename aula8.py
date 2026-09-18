# ══════════════════════════════════════════════════════════════
# ATIVIDADE 8 — LIMPEZA E PREPARAÇÃO DA BASE
# Projeto: [Spam Blocker] Aluno: [Lucas da Silva Vargas]
# ══════════════════════════════════════════════════════════════

# ── 1. IMPORTAÇÕES ────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from urllib.request import urlopen
import zipfile
import io
#from deep_translator import GoogleTranslator

# ── 2. CARREGAMENTO ───────────────────────────────────────────
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
df_original = df.copy() # guarda o original

# ── 3. INSPEÇÃO INICIAL ───────────────────────────────────────
print(df.shape)
#(5572, 2)
df.info()
'''
    <class 'pandas.DataFrame'>
    RangeIndex: 5572 entries, 0 to 5571
    Data columns (total 2 columns):
    #   Column   Non-Null Count  Dtype
    ---  ------   --------------  -----
    0   label    5572 non-null   str  
    1   message  5572 non-null   str  
    dtypes: str(2)
'''

print(df.isnull().sum())
'''
    label      0
    message    0
'''
print(df.duplicated().sum())
#403

# ── 4. TRATAMENTO DE AUSENTES ─────────────────────────────────
#Não se aplica.

# ── 5. DUPLICIDADES ───────────────────────────────────────────
df = df.drop_duplicates() #remove as 403 duplicadas. Já que a base é grande, da pra se dar o luxo
print("Linhas restantes:", len(df))
#Linhas restantes: 5169

# ── 6. PADRONIZAÇÃO DE CATEGORIAS ─────────────────────────────
#Não se aplica


# ── 7. CODIFICAÇÃO ────────────────────────────────────────────
print(df.label)
'''
    0        ham
    1        ham
    2       spam
    3        ham
    4        ham
            ... 
'''
mapa_ordinal = {
    "ham": 0,
    "spam": 1
}
df.label = df.label.map(mapa_ordinal)
print(df.label.isnull().sum())
# 0

print(df.label)
'''
    0       0
    1       0
    2       1
    3       0
    4       0
        ...
'''
# ── 8. ESCALA (se necessário para o seu modelo) ───────────────
# não se aplica

# ── 9. SALVAMENTO ────────────────────────────────────────────
df.to_csv("dados_tratados.csv", index=False)
print("Concluído. Linhas finais:", len(df))