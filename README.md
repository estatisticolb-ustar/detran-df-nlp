# Análise de Sentimento e Classificação de Manifestações dos DETRANs Brasileiros

**PLN aplicado ao Setor Público — Dados Reais da CGU (2022–2024)**

Projeto Final · Disciplina de Deep Learning e Processamento de Linguagem Natural · **Modalidade 2 (Dupla)**
Integrantes: **Liandro S. Barcellos · Rodrigo A. da Costa** · Prof. Danny

---

## Visão geral

Este projeto aplica técnicas de Processamento de Linguagem Natural a manifestações de cidadãos sobre os DETRANs, coletadas via dados abertos do portal **Fala.BR (CGU)**. O objetivo é responder: *quais são os principais motivos de insatisfação dos cidadãos com os DETRANs, e como o sentimento varia por tipo de serviço e ao longo do tempo?*

O pipeline combina **análise de sentimento** (DistilBERT multilíngue) e **classificação textual** (TF-IDF + SVM / Naive Bayes / Random Forest) sobre **9.513 manifestações** de DETRANs de **8 estados** no período **2022–2024**.

> O relatório técnico completo está em [`RELATORIO.md`](RELATORIO.md). A apresentação está em `Apresentacao_DETRAN_NLP.pdf`.

---

## Descrição da coleta de dados

Esta seção atende ao entregável obrigatório da Modalidade 2 (de onde os dados foram coletados, período, filtros e limitações).

- **Fonte:** portal **Fala.BR** — Plataforma Integrada de Ouvidoria e Acesso à Informação da Controladoria-Geral da União (CGU), via coleta automatizada de dados abertos. **Nenhuma base pronta ou plataforma como Kaggle foi utilizada** — a coleta é inédita e feita pelo próprio grupo.
- **Período analisado:** 01/01/2022 a 31/12/2024 (3 anos).
- **Filtros aplicados:**
  - Órgão de trânsito (nome "DETRAN").
  - Recorte temporal acima.
  - No pré-processamento, descarte de registros sem texto ou com texto com menos de 20 caracteres.
- **Escopo geográfico:** o **DETRAN-DF não aderiu ao Fala.BR federal** no período, então o escopo foi ampliado para **8 estados** (RN, TO, RJ, MS, PI, AM, RR, RO), preservando o domínio de trânsito.
- **Volume final:** 9.513 manifestações.

**Distribuição por UF do órgão:**

| UF | Manifestações |
|---|---:|
| RN — Rio Grande do Norte | 3.783 |
| TO — Tocantins | 2.034 |
| RJ — Rio de Janeiro | 1.624 |
| MS — Mato Grosso do Sul | 776 |
| PI — Piauí | 584 |
| Demais (AM, RR, RO) | 712 |
| **Total** | **9.513** |

**Tipos de manifestação:** Reclamação (4.165), Solicitação (3.724), Comunicação (791), Denúncia (524), Elogio (229), Sugestão (80).

**Limitações da base:** os registros da CGU privilegiam metadados (assunto, tipo, serviço, situação) e os textos livres são curtos (média ~2 palavras úteis após limpeza), o que restringe a aplicação direta de modelos contextuais a textos longos. Não há anotação manual de sentimento, o que impede avaliação supervisionada direta da qualidade do sentimento previsto.

---

## Estrutura do repositório

```
detran-df-nlp/
├── scripts/
│   ├── 01_coleta_falabr.py        # Coleta via Fala.BR (CGU)
│   ├── 02_preprocessamento.py     # Limpeza, anonimização, tokenização, categorias
│   ├── 03_analise_sentimento.py   # DistilBERT multilíngue (HuggingFace)
│   ├── 04_classificacao.py        # TF-IDF + SVM / Naive Bayes / Random Forest
│   └── 05_eda_visualizacoes.py    # Gráficos, heatmaps e wordclouds
├── data/
│   ├── raw/                       # CSV bruto da coleta (gerado pelo script 01)
│   └── processed/                 # CSVs processados (gerados pelos scripts 02 e 03)
├── outputs/
│   ├── figures/                   # Imagens geradas
│   └── relatorio_metricas.txt     # Métricas de classificação
├── Apresentacao_DETRAN_NLP.pdf    # Slides (entregável)
├── RELATORIO.md                   # Relatório técnico completo
├── requirements.txt
└── README.md
```

> As pastas `data/` e `outputs/` são criadas automaticamente pelos scripts na primeira execução.

---

## Como executar

Requer **Python 3.10+**.

```bash
# 1. Clonar e instalar dependências
git clone https://github.com/estatisticolb-ustar/detran-df-nlp.git
cd detran-df-nlp
pip install -r requirements.txt

# 2. Rodar o pipeline na ordem (cada script consome a saída do anterior)
python scripts/01_coleta_falabr.py        # gera data/raw/manifestacoes_raw.csv
python scripts/02_preprocessamento.py     # gera data/processed/manifestacoes_processadas.csv
python scripts/03_analise_sentimento.py   # gera data/processed/manifestacoes_sentimento.csv + figuras
python scripts/04_classificacao.py        # gera outputs/relatorio_metricas.txt + matriz de confusão
python scripts/05_eda_visualizacoes.py    # gera figuras de EDA e wordclouds
```

> Na primeira execução, o script 02 baixa automaticamente os recursos do NLTK (`stopwords`, `punkt`) e o script 03 baixa o modelo do HuggingFace.

---

## Metodologia (resumo)

| Etapa | Técnica |
|---|---|
| Coleta | API de dados abertos Fala.BR / CGU |
| Pré-processamento | Limpeza, anonimização (CPF/e-mail), stopwords NLTK, tokenização |
| Sentimento | DistilBERT multilíngue (`lxyuan/distilbert-base-multilingual-cased-sentiments-student`) |
| Classificação | TF-IDF (1–2 gramas) + LinearSVC / MultinomialNB / RandomForest |
| Visualização | matplotlib, seaborn, wordcloud |

Detalhes, decisões metodológicas e leitura crítica dos resultados em [`RELATORIO.md`](RELATORIO.md).

---

## Principais achados

- **Sentimento:** 49,3% negativo, 48,3% neutro, 2,4% positivo.
- **Atendimento** é a categoria mais crítica (58,3% das reclamações negativas).
- **Sistema digital** é o segundo ponto crítico (47,4% de reclamações).
- RN e TO concentram ~60% do volume.

> **Nota sobre as métricas de classificação:** os valores próximos de 1,000 decorrem de vazamento entre a regra de rotulagem (categorias derivadas do "assunto") e as features TF-IDF; devem ser lidos como verificação de consistência do pipeline, não como capacidade preditiva. Ver Seção 7 do relatório.

---

## Referências

Listadas integralmente em [`RELATORIO.md`](RELATORIO.md) — 5 sobre o domínio (setor público/ouvidoria) e 5 sobre as técnicas (BERT, DistilBERT, BERTimbau, análise de sentimento e classificação textual).
