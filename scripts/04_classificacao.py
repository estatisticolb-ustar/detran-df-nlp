import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings; warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_PATH = os.path.join(BASE_DIR,"data","processed","manifestacoes_sentimento.csv")
FIG_DIR = os.path.join(BASE_DIR,"outputs","figures")
RELATORIO = os.path.join(BASE_DIR,"outputs","relatorio_metricas.txt")
os.makedirs(FIG_DIR,exist_ok=True)

TFIDF = {"ngram_range":(1,2),"max_features":15000,"sublinear_tf":True,"min_df":2}

pipelines = {
    "SVM (LinearSVC)": Pipeline([("tfidf",TfidfVectorizer(**TFIDF)),("clf",LinearSVC(C=1.0,max_iter=2000))]),
    "Naive Bayes": Pipeline([("tfidf",TfidfVectorizer(**TFIDF)),("clf",MultinomialNB(alpha=0.1))]),
    "Random Forest": Pipeline([("tfidf",TfidfVectorizer(**TFIDF,max_features=5000)),("clf",RandomForestClassifier(n_estimators=100,random_state=42))])
}

if __name__=="__main__":
    df=pd.read_csv(INPUT_PATH,encoding="utf-8-sig").dropna(subset=["texto_tokens","categoria"])
    cont=df["categoria"].value_counts()
    df=df[df["categoria"].isin(cont[cont>=10].index)].copy()
    X=df["texto_tokens"].values; y=df["categoria"].values
    labels=sorted(df["categoria"].unique().tolist())
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    resultados=[]; relat=[]
    for nome,pipe in pipelines.items():
        pipe.fit(Xtr,ytr); yp=pipe.predict(Xte)
        acc=accuracy_score(yte,yp); f1=f1_score(yte,yp,average="macro",zero_division=0)
        resultados.append({"Modelo":nome,"Accuracy":round(acc,4),"F1-macro":round(f1,4)})
        relat.append(f"\n{'='*60}\nModelo: {nome}\nAccuracy: {acc:.4f} | F1-macro: {f1:.4f}\n\n{classification_report(yte,yp,zero_division=0)}")
        print(f"{nome} - Acc: {acc:.3f} F1: {f1:.3f}")
    open(RELATORIO,"w",encoding="utf-8").write("\n".join(relat))
    dfr=pd.DataFrame(resultados)
    melhor=dfr["F1-macro"].idxmax()
    nm=dfr.loc[melhor,"Modelo"]
    print(f"\nMelhor modelo: {nm}")
    print(dfr.to_string(index=False))
    yp=pipelines[nm].predict(Xte)
    cm=confusion_matrix(yte,yp,labels=labels)
    fig,ax=plt.subplots(figsize=(9,7))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=labels,yticklabels=labels,ax=ax)
    ax.set_title(f"Matriz de Confusao - {nm}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"matriz_confusao.png"),dpi=150)
    plt.close()
    print("Concluido.")
