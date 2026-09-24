# Spam Blocker
Projeto de Lucas da Silva Vargas — Fundamentos de Inteligência Artificial / ADS.
Problema e público: preparar a classificação de SMS legítimos e spam para apoiar usuários de telefonia móvel na identificação de mensagens indesejadas. É um projeto acadêmico, ainda sem classificador treinado.

## Fonte e licença
SMS Spam Collection — Almeida, T. & Hidalgo, J. (2011), UCI: https://doi.org/10.24432/C5CC84.
Licença da base: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), conforme a [página da UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
Original preservado: [sms_original.zip](sms_original.zip). A versão tratada modifica espaços, pontuação, duplicatas e rótulos; atribuição e licença referem-se à base, sem atribuir uma licença nova ao código.

## Arquivos e ordem de execução
- [Atividade8_Preparacao.ipynb](Atividade8_Preparacao.ipynb): preparação final executada, com inspeção, justificativas, decisões e verificações; começar por ele.
- [aula8.py](aula8.py): as mesmas células de código para execução como script, alternativa ao notebook.
- [dados_tratados.csv](dados_tratados.csv): saída final verificada, com `label` (0 = ham, 1 = spam) e `message`.
- [registro_decisoes.csv](registro_decisoes.csv): transformação, coluna, motivo, impacto e risco de cada ação.
- [erros.py](erros.py): diagnóstico histórico do CSV derivado [spam.csv](spam.csv), ambos preservados; não usar esse CSV no treinamento.
- [grafico.py](grafico.py): exploração histórica da frequência das classes e do comprimento das mensagens; não é a rotina final de preparação.

## Reprodução
1. Instalar Python 3 e pandas: `pip install pandas`. Para notebook local, instalar também `notebook`.
2. Baixar o notebook e o ZIP para a mesma pasta; abrir no Jupyter. No Colab, enviar ambos; sem o ZIP, a rotina baixa a UCI automaticamente.
3. Reiniciar e executar todas as células, sem erros, e salvar com as saídas. Alternativa: `python aula8.py` dentro da pasta dos arquivos, sem `-O`.
4. Serão gerados os CSVs tratado e de decisões. O original não é sobrescrito; a leitura de volta e seu SHA-256 são conferidos.

## Evolução e prontidão
Leitura corrigida com `QUOTE_NONE`: 5.574 registros reais; o leitor anterior unia três SMS e retornava 5.572 linhas. A conferência compara cada linha com a fonte.
Deduplicação inicial: 403 linhas removidas. Espaços normalizados em 407 mensagens. Reparo restrito de cinco caracteres de pontuação C1. Nova deduplicação: 12 cópias spam removidas (11 após espaços e 1 adicional após o reparo de pontuação).
Base final: **5.159 linhas, 2 colunas, zero ausentes e zero duplicatas**; 4.518 ham (87,58%) e 641 spam (12,42%).
Nenhuma operação aprende parâmetros. As features contêm apenas `message`, sem IDs ou colunas pós-desfecho; `label` é somente alvo. Na próxima etapa, dividir com estratificação e ajustar TF-IDF apenas no treino, aplicando `transform` no teste.

## Uso de IA e entrega
ChatGPT apoiou a revisão, o código complementar, a execução de conferência e a documentação; o estudante deve compreender e revisar as decisões. O notebook explica `assert` e `pd.testing.assert_frame_equal`.
O PDF final do modelo padrão é entregue pela plataforma Aula, junto ao link da raiz deste repositório. Conferir os links sem autenticação antes do envio.
