# Atividade 8 — Entrega final | Lucas da Silva Vargas | Spam Blocker
# Código equivalente às células do notebook. Execute sem python -O.

from pathlib import Path
from urllib.request import urlopen
import hashlib
import zipfile
import io
import csv
import pandas as pd

origem = Path("sms_original.zip")
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
if not origem.exists():
    with urlopen(url, timeout=60) as resposta:
        origem.write_bytes(resposta.read())
conteudo = origem.read_bytes()
hash_original = hashlib.sha256(conteudo).hexdigest()
with zipfile.ZipFile(io.BytesIO(conteudo)) as pacote:
    texto = pacote.read("SMSSpamCollection").decode("utf-8", errors="strict")
print("SHA-256 do original:", hash_original)
print("Pandas:", pd.__version__)


leitura_esboco = pd.read_csv(io.StringIO(texto), sep="\t", header=None,
                             names=["label", "message"])
df = pd.read_csv(io.StringIO(texto), sep="\t", header=None,
                 names=["label", "message"], quoting=csv.QUOTE_NONE)
linhas_fonte = texto.splitlines()
assert all(linha.startswith(("ham\t", "spam\t")) for linha in linhas_fonte)
referencia = pd.DataFrame([linha.split("\t", 1) for linha in linhas_fonte],
                         columns=["label", "message"])
# A comparação garante que cada SMS foi carregado sem unir registros.
pd.testing.assert_frame_equal(df, referencia, check_dtype=False)
df_original = df.copy(deep=True)
print("Leitura do esboço:", leitura_esboco.shape)
print("Leitura final:", df.shape)
print("Linhas físicas na fonte:", len(linhas_fonte))
print("Registros com quebra de linha na leitura anterior:",
      int(leitura_esboco["message"].str.contains("\n").sum()))
print("Tipos:\n" + df.dtypes.to_string())
print("Ausentes:\n" + df.isna().sum().to_string())
print("Duplicatas exatas:", int(df.duplicated().sum()))
print("Classes iniciais:\n" + pd.DataFrame({
    "quantidade": df["label"].value_counts(),
    "percentual": df["label"].value_counts(normalize=True).mul(100).round(2)
}).to_string())
assert set(df["label"].unique()) == {"ham", "spam"}
assert df.isna().sum().sum() == 0
assert df["message"].str.strip().ne("").all()
decisoes = []
def registrar(acao, coluna, motivo, impacto, risco):
    decisoes.append({"transformacao": acao, "coluna_afetada": coluna,
                     "motivo": motivo, "impacto": impacto, "risco": risco})
registrar("Leitura TSV sem interpretação de aspas", "label e message",
          "Preservar cada linha canônica; evitar reconstruir o CSV fragmentado",
          f"Leitura: {len(leitura_esboco)} -> {len(df)} linhas; 2 colunas; original intacto",
          "Mudança futura de formato; comparação linha a linha valida esta leitura")


antes = len(df)
duplicatas_iniciais = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
print("Deduplicação inicial:", antes, "->", len(df))
print("Removidas:", duplicatas_iniciais)
registrar("Deduplicação inicial", "label e message", "Evitar peso adicional de cópias exatas",
          f"{duplicatas_iniciais} linhas removidas; {antes} -> {len(df)}; 0 colunas removidas",
          "Altera frequências; mensagens semelhantes ainda podem existir")


normalizado = df["message"].str.replace(r"\s+", " ", regex=True).str.strip()
linhas_espacos = int(normalizado.ne(df["message"]).sum())
df["message"] = normalizado
assert df["message"].ne("").all()
print("Mensagens com espaços normalizados:", linhas_espacos)
print("Duplicatas após espaços:", int(df.duplicated().sum()))
registrar("Normalização de espaços", "message", "Reduzir variações de formatação preservando palavras",
          f"{linhas_espacos} mensagens alteradas; {len(df)} linhas antes e depois",
          "Perda de sinais de estilo; pode revelar novas duplicatas")


mapa_pontuacao = {0x91: "‘", 0x92: "’", 0x93: "“", 0x94: "”", 0x96: "–"}
reparado = df["message"].str.translate(mapa_pontuacao)
linhas_pontuacao = int(reparado.ne(df["message"]).sum())
caracteres_pontuacao = sum(sum(ord(c) in mapa_pontuacao for c in t) for t in df["message"])
df["message"] = reparado
print("Mensagens com pontuação reparada:", linhas_pontuacao)
print("Caracteres reparados:", caracteres_pontuacao)
print("U+FFFD restante:", int(df["message"].str.contains("\ufffd", regex=False).sum()))
print("Controles C1 restantes:", int(df["message"].str.contains(r"[\x80-\x9f]", regex=True).sum()))
assert not df["message"].str.contains("\ufffd", regex=False).any()
assert not df["message"].str.contains(r"[\x80-\x9f]", regex=True).any()
registrar("Reparo de cinco caracteres C1", "message", "Corrigir pontuação inspecionada sem recodificar todo o texto",
          f"{linhas_pontuacao} mensagens e {caracteres_pontuacao} caracteres; 0 linhas removidas",
          "Interpretação indevida de caractere; mapa limitado aos casos inspecionados")
conflitos = int(df.groupby("message")["label"].nunique().gt(1).sum())
assert conflitos == 0, "Há textos iguais com rótulos diferentes: revisar manualmente"
antes = len(df)
novas_duplicatas = int(df.duplicated().sum())
classes_removidas = df.loc[df.duplicated(), "label"].value_counts().to_dict()
df = df.drop_duplicates().reset_index(drop=True)
print("Rótulos conflitantes:", conflitos)
print("Nova deduplicação:", antes, "->", len(df))
print("Classes das cópias removidas:", classes_removidas)
registrar("Deduplicação após normalização", "label e message", "Retirar cópias reveladas pela normalização",
          f"{novas_duplicatas} linhas removidas; {antes} -> {len(df)}; classes: {classes_removidas}",
          "Nova alteração das proporções; remoção limitada a pares exatamente iguais")


df["label"] = df["label"].map({"ham": 0, "spam": 1})
assert df["label"].notna().all()
assert set(df["label"].unique()) == {0, 1}
print("Rótulos codificados:", len(df))
print("Tipo do alvo:", df["label"].dtype)
registrar("Codificação binária fixa", "label", "Representar duas classes com um mapa fixo",
          f"{len(df)} valores em 1 coluna; 0 linhas removidas",
          "Categoria desconhecida virar nulo; validação interrompe o processo")
print("Classes finais:\n" + pd.DataFrame({
    "quantidade": df["label"].value_counts(),
    "percentual": df["label"].value_counts(normalize=True).mul(100).round(2)
}).to_string())


X = df[["message"]].copy()
y = df["label"].copy()
assert list(X.columns) == ["message"]
assert "label" not in X.columns
assert not df["message"].duplicated().any()
print("Features:", list(X.columns), "| Alvo:", y.name)
print("Identificadores entre as features: 0")
print("Colunas pós-desfecho entre as features: 0")
print("Transformações ajustadas por fit: nenhuma")
print("Etapa seguinte: divisão estratificada, TF-IDF somente no treino e treinamento.")


registro = pd.DataFrame(decisoes)
registro.to_csv("registro_decisoes.csv", index=False)
print(registro.to_string(index=False))
df.to_csv("dados_tratados.csv", index=False)
conferencia = pd.read_csv("dados_tratados.csv")
pd.testing.assert_frame_equal(df, conferencia, check_dtype=False)
assert conferencia.shape == df.shape
assert conferencia.isna().sum().sum() == 0
assert not conferencia.duplicated().any()
assert conferencia["message"].str.strip().ne("").all()
assert hashlib.sha256(origem.read_bytes()).hexdigest() == hash_original
print("Dimensões:", df_original.shape, "->", conferencia.shape)
print("Linhas removidas:", len(df_original) - len(conferencia))
print("Colunas removidas: 0")
print("Ausentes e duplicatas finais: 0 e 0")
print("CSV recarregado e conferido: OK | Original preservado: OK")
