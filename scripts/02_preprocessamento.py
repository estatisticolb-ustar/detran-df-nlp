"""
Script 02 - Pre-processamento
"""
import pandas as pd
import re
import os
import nltk
from unidecode import unidecode
for recurso in ["stopwords", "punkt", "punkt_tab"]:
    nltk.download(recurso, quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_PATH = os.path.join(BASE_DIR,"data","raw","manifestacoes_raw.csv")
OUTPUT_PATH = os.path.join(BASE_DIR,"data","processed","manifestacoes_processadas.csv")
STOP_PT = set(stopwords.words("portuguese"))
STOP_PT.update({"detran","df","brasilia","gdf","distrito","federal","senhor","senhora","prezado","prezada","solicito","venho","informo"})
def limpar_texto(t):
    if not isinstance(t,str) or not t.strip(): return ""
    t=t.lower()
    t=re.sub(r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}','[CPF]',t)
    t=re.sub(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}','[EMAIL]',t)
    t=re.sub(r'https?://\S+|www\.\S+','',t)
    t=re.sub(r'[^\w\s]',' ',t)
    t=re.sub(r'\s+',' ',t).strip()
    return t
def tokenizar(t):
    if not t: return ""
    tks=word_tokenize(t,language="portuguese")
    return " ".join([x for x in tks if x.isalpha() and x not in STOP_PT and len(x)>2])
def categoria(a):
    a=str(a).lower()
    if any(p in a for p in ["habilitacao","cnh","carteira","condutor","exame"]): return "habilitacao"
    if any(p in a for p in ["veiculo","emplacamento","licenciamento","crv","crlv"]): return "veiculo"
    if any(p in a for p in ["multa","infracao","auto","penalidade","recurso"]): return "infracoes"
    if any(p in a for p in ["atendimento","servico","agendamento","prazo","demora"]): return "atendimento"
    if any(p in a for p in ["sistema","site","portal","digital","online","app"]): return "sistema_digital"
    return "outros"
if __name__=="__main__":
    df=pd.read_csv(INPUT_PATH,encoding="utf-8-sig")
    df=df.dropna(subset=["texto"])
    df=df[df["texto"].str.strip().str.len()>20].copy()
    df["texto_limpo"]=df["texto"].apply(limpar_texto)
    df["texto_tokens"]=df["texto_limpo"].apply(tokenizar)
    df["categoria"]=df["assunto"].apply(categoria)
    df["data_registro"]=pd.to_datetime(df["data_registro"],errors="coerce")
    df["num_palavras"]=df["texto_limpo"].apply(lambda x:len(x.split()))
    os.makedirs(os.path.dirname(OUTPUT_PATH),exist_ok=True)
    df.to_csv(OUTPUT_PATH,index=False,encoding="utf-8-sig")
    print(f"Salvo: {OUTPUT_PATH} | {len(df)} registros")
    print(df["categoria"].value_counts())
