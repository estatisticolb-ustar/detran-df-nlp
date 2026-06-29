"""
Script 01 – Coleta de manifestações do Detran-DF via portal Fala.BR
Fonte: https://falabr.cgu.gov.br
Documentação da API: https://falabr.cgu.gov.br/swagger-ui/index.html

Como executar:
    pip install requests pandas
    python 01_coleta_falabr.py
"""

import requests
import pandas as pd
import time
import os

BASE_URL = "https://falabr.cgu.gov.br/api/v1/manifestacoes"
HEADERS = {"Accept": "application/json", "User-Agent": "pesquisa-academica-unb-2025"}
SIORG_DETRAN_DF = "57877"
PARAMS_BASE = {"codigoSiorgOrgaoDestinatario": SIORG_DETRAN_DF, "dataInicio": "2022-01-01", "dataFim": "2024-12-31", "quantidade": 100, "pagina": 1, "situacao": ""}
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "manifestacoes_raw.csv")

def coletar_pagina(pagina, params):
    p = {**params, "pagina": pagina}
    try:
        r = requests.get(BASE_URL, headers=HEADERS, params=p, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Erro: {e}")
        return {}

def extrair_campos(item):
    return {"id": item.get("id",""), "data_registro": item.get("dataRegistro",""), "tipo": item.get("tipoManifestacao",{}).get("descricao",""), "assunto": item.get("assunto",{}).get("descricao",""), "texto": item.get("descricao",""), "situacao": item.get("situacao",{}).get("descricao",""), "uf_solicitante": item.get("ufSolicitante",""), "municipio": item.get("municipio",{}).get("nome","")}

def coletar_tudo(params, max_paginas=50):
    registros = []
    for pagina in range(1, max_paginas+1):
        print(f"Coletando página {pagina}...", end=" ")
        dados = coletar_pagina(pagina, params)
        if not dados: break
        itens = dados.get("data", dados.get("content", dados.get("registros",[])))
        if not itens: break
        for item in itens: registros.append(extrair_campos(item))
        print(f"{len(itens)} registros")
        if pagina >= dados.get("totalPaginas",1): break
        time.sleep(1)
    return pd.DataFrame(registros)

if __name__ == "__main__":
    df = coletar_tudo(PARAMS_BASE)
    if not df.empty:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
        print(f"Salvo: {OUTPUT_PATH} | {len(df)} registros")
