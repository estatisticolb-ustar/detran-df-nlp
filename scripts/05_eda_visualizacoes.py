import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import warnings; warnings.filterwarnings("ignore")
try:
    from wordcloud import WordCloud; WC_OK=True
except: WC_OK=False

BASE_DIR=os.path.dirname(os.path.dirname(__file__))
INPUT_PATH=os.path.join(BASE_DIR,"data","processed","manifestacoes_sentimento.csv")
FIG_DIR=os.path.join(BASE_DIR,"outputs","figures")
os.makedirs(FIG_DIR,exist_ok=True)

CORES={"habilitacao":"#3498db","veiculo":"#e67e22","infracoes":"#e74c3c","atendimento":"#9b59b6","sistema_digital":"#1abc9c","outros":"#95a5a6"}

def plot_categoria(df):
    c=df["categoria"].value_counts()
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(c.index,c.values,color=[CORES.get(x,"#7f8c8d") for x in c.index],edgecolor="white",width=0.6)
    ax.set_title("Volume por Categoria - Detran-DF",fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"eda_volume_categoria.png"),dpi=150)
    plt.close(); print("Salvo: eda_volume_categoria.png")

def plot_mensal(df):
    if "data_registro" not in df.columns: return
    d2=df.copy()
    d2["mes"]=pd.to_datetime(d2["data_registro"],errors="coerce").dt.to_period("M")
    m=d2.groupby("mes").size().reset_index(name="total")
    m["m"]=m["mes"].astype(str)
    fig,ax=plt.subplots(figsize=(12,4))
    ax.bar(m["m"],m["total"],color="#3498db",edgecolor="white",width=0.8)
    ax.set_title("Volume por Mes - Detran-DF",fontsize=13)
    plt.xticks(rotation=45,ha="right",fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"eda_volume_mensal.png"),dpi=150)
    plt.close(); print("Salvo: eda_volume_mensal.png")

def plot_palavras(df):
    c=df["num_palavras"].dropna()
    fig,ax=plt.subplots(figsize=(9,4))
    ax.hist(c,bins=40,color="#3498db",edgecolor="white")
    ax.axvline(c.median(),color="#e74c3c",linestyle="--",label=f"Mediana: {c.median():.0f} palavras")
    ax.set_title("Comprimento dos Textos",fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"eda_palavras.png"),dpi=150)
    plt.close(); print("Salvo: eda_palavras.png")

def wordcloud(df,sent,cor):
    if not WC_OK: return
    t=df[df["sentimento"]==sent]["texto_tokens"].dropna()
    if t.empty: return
    wc=WordCloud(width=800,height=400,background_color="white",colormap=cor,max_words=80,collocations=False).generate(" ".join(t.tolist()))
    fig,ax=plt.subplots(figsize=(10,5))
    ax.imshow(wc,interpolation="bilinear"); ax.axis("off")
    ax.set_title(f"Palavras frequentes - {sent.upper()}",fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,f"wordcloud_{sent}.png"),dpi=150)
    plt.close(); print(f"Salvo: wordcloud_{sent}.png")

if __name__=="__main__":
    df=pd.read_csv(INPUT_PATH,encoding="utf-8-sig")
    print(f"{len(df)} registros")
    plot_categoria(df); plot_mensal(df); plot_palavras(df)
    wordcloud(df,"negativo","Reds"); wordcloud(df,"positivo","Greens"); wordcloud(df,"neutro","Blues")
    print("Concluido.")
