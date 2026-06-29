"""
Script 03 - Analise de Sentimento
import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_PATH  = os.path.join(BASE_DIR,"data","processed","manifestacoes_processadas.csv")
OUTPUT_CSV  = os.path.join(BASE_DIR,"data","processed","manifestacoes_sentimento.csv")
FIG_DIR     = os.path.join(BASE_DIR,"outputs","figures")
os.makedirs(FIG_DIR,exist_ok=True)

def carregar_modelo():
    from transformers import pipeline
    print("Carregando modelo: lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    return pipeline("text-classification",model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",device=-1,truncation=True,max_length=512)

def prever_sentimento(texto,classifier):
    if not isinstance(texto,str) or len(texto.strip())<5: return "neutro",0.0
    try:
        r=classifier(texto[:512])[0]
        l=r["label"].lower()
        mapa={"positive":"positivo","negative":"negativo","neutral":"neutro"}
        return mapa.get(l,l),round(r["score"],4)
    except: return "neutro",0.0

def plot_sentimento(df):
    contagem=df["sentimento"].value_counts()
    cores={"positivo":"#2ecc71","neutro":"#95a5a6","negativo":"#e74c3c"}
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(contagem.index,contagem.values,color=[cores.get(s,"#7f8c8d") for s in contagem.index],edgecolor="white",width=0.55)
    ax.set_title("Distribuicao de Sentimento - Detran-DF",fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"distribuicao_sentimento.png"),dpi=150)
    plt.close()

def plot_heatmap(df):
    t=df.groupby(["categoria","sentimento"]).size().unstack(fill_value=0)
    t = t.div(t.sum(axis=1),axis=0)*100
    fig,ax=plt.subplots(figsize=(10,6))
    sns.heatmap(t,annot=True,fmt=".1f",cmap="RdYlGn",ax=ax)
    ax.set_title("Sentimento por Categoria - Detran-DF (%)",fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"sentimento_por_categoria.png"),dpi=150)
    plt.close()

if __name__=="__main__":
    df=pd.read_csv(INPUT_PATH,encoding="utf-8-sig")
    classifier=carregar_modelo()
    sentimentos,scores=[],[]
    for texto in tqdm(df["texto_limpo"].fillna(""),desc="Sentimento"):
        s,c=prever_sentimento(texto,classifier)
        sentimentos.append(s);scores.append(c)
    df["sentimento"]=sentimentos
    df["sentimento_score"]=scores
    df.to_csv(OUTPUT_CSV,index=False,encoding="utf-8-sig")
    print(df["sentimento"].value_counts())
    plot_sentimento(df)
    plot_heatmap(df)
    print("Concluido.")
