import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.dummy import DummyClassifier

df = pd.read_csv("dados_tratados.csv")

X = df["message"]
y = df["label"]

#80/20
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


vetorizador = CountVectorizer()
X_treino_vetorizado = vetorizador.fit_transform(X_treino)
X_teste_vetorizado = vetorizador.transform(X_teste)

#Gaussian
X_treino_vetorizado_array = X_treino_vetorizado.toarray() 
X_teste_vetorizado_array  = X_teste_vetorizado.toarray()

print(X_teste_vetorizado[0])

#----Dummy----
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_clf.fit(X_treino, y_treino)
y_pred_baseline = dummy_clf.predict(X_teste)
accuracy_baseline = accuracy_score(y_teste, y_pred_baseline)
print(f"Acurácia do Modelo Baseline (classificador 'chutando' a classe mais frequente): {accuracy_baseline*100:.4f}%") #87.60%
#-------------

#modelo = GaussianNB()  #89%
modelo = MultinomialNB() #98%

modelo.fit(X_treino_vetorizado, y_treino) #multinomial
#modelo.fit(X_treino_vetorizado_array, y_treino) #Gaussian

previsoes = modelo.predict(X_teste_vetorizado)
#previsoes = modelo.predict(X_teste_vetorizado_array) #Gaussian
acuracia = accuracy_score(y_teste, previsoes)

print(f"Acurácia do modelo: {acuracia*100:.4f}%")
print("\nRelatório de classificação:")
print(classification_report(y_teste, previsoes))

'''
mensagem = input("\nDigite uma mensagem para testar: ")

mensagem_vetorizada = vetorizador.transform([mensagem])
resultado = modelo.predict(mensagem_vetorizada)[0]

if resultado == 1:
    print("Resultado: SPAM")
else:
    print("Resultado: HAM")
'''