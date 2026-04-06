"""
app.py — Interface web do Leitor de Gabaritos CCF
Rode localmente:  streamlit run app.py
Deploy:           Streamlit Cloud (streamlit.io)
"""

import streamlit as st
import tempfile
import os
import csv
import io
import time
import threading
from datetime import datetime

import google.generativeai as genai
from PIL import Image

from services.omr_processor import process_answer_sheet

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Leitor de Gabaritos CCF",
    page_icon="📝",
    layout="centered",
)

# ── Chave API (Streamlit Secrets ou variável de ambiente) ─────────────────────
def get_api_keys():
    keys = []
    # Streamlit Cloud: secrets configurados no painel do Streamlit
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if k.lower().startswith("google") and str(v).strip():
                keys.append(str(v).strip())
        if not keys and "GEMINI_API_KEY" in st.secrets:
            keys.append(str(st.secrets["GEMINI_API_KEY"]).strip())
    # Fallback: variáveis de ambiente locais
    if not keys:
        for k, v in os.environ.items():
            if k.lower().startswith("google") and v.strip():
                keys.append(v.strip())
        if not keys:
            v = os.getenv("GEMINI_API_KEY", "")
            if v.strip():
                keys.append(v.strip())
    return keys

API_KEYS = get_api_keys()
_key_lock = threading.Lock()
_key_idx = 0

def _next_key():
    global _key_idx
    with _key_lock:
        _key_idx = (_key_idx + 1) % len(API_KEYS)

def _get_key():
    with _key_lock:
        return API_KEYS[_key_idx]

def ler_nome_via_gemini(image_path: str) -> str:
    if not API_KEYS:
        return ""
    for _ in range(len(API_KEYS) * 2):
        try:
            genai.configure(api_key=_get_key())
            model = genai.GenerativeModel("gemini-2.5-flash")
            img = Image.open(image_path)
            img.thumbnail((800, 800))
            prompt = (
                "Esta é uma folha de resposta escolar. No cabeçalho há um campo 'ALUNO' "
                "preenchido pelo aluno com letra de forma em caixa alta. "
                "Leia o nome escrito nesse campo e responda APENAS com o nome, "
                "sem pontuação, sem explicação, sem aspas. "
                "Se não conseguir ler, responda apenas: (ilegível)"
            )
            resp = model.generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(temperature=0.0),
            )
            return resp.text.strip().strip("\"'")
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "exhausted" in err or "503" in err:
                _next_key()
                time.sleep(1)
            else:
                return ""
    return ""


def processar_arquivo(uploaded_file) -> dict:
    """Salva o arquivo em temp, roda OMR + leitura de nome, retorna dict de resultado."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        resultado = process_answer_sheet(tmp_path)

        if "erro" in resultado:
            return {"arquivo": uploaded_file.name, "erro": resultado["erro"],
                    "nome": "", "obj": {}, "soma": {}}

        obj_raw  = resultado.get("objetivas", {})
        soma_raw = resultado.get("somatorias", {})
        nome     = ler_nome_via_gemini(tmp_path)

        return {
            "arquivo": uploaded_file.name,
            "nome":    nome,
            "obj":     {str(k): v for k, v in obj_raw.items()  if v is not None},
            "soma":    {str(k): v for k, v in soma_raw.items() if v is not None},
            "erro":    "",
        }
    finally:
        os.unlink(tmp_path)


def gerar_csv(resultados: list) -> str:
    obj_q  = ["3", "4", "5", "6", "7", "8"]
    soma_q = ["9", "10"]
    headers = ["Arquivo", "NomeAluno"] + [f"Q{q}" for q in obj_q] + \
              [f"SOMA_{q}" for q in soma_q] + ["Erro"]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, delimiter=";")
    writer.writeheader()
    for r in resultados:
        row = {
            "Arquivo":   r["arquivo"],
            "NomeAluno": r["nome"],
            "Erro":      r["erro"],
        }
        for q in obj_q:  row[f"Q{q}"]    = r["obj"].get(q, "")
        for q in soma_q: row[f"SOMA_{q}"] = r["soma"].get(q, "")
        writer.writerow(row)
    return buf.getvalue()


# ── Interface ─────────────────────────────────────────────────────────────────
st.title("📝 Leitor de Gabaritos CCF")
st.caption("OMR local + leitura de nome via Gemini")

if not API_KEYS:
    st.warning(
        "Nenhuma chave API configurada. O nome dos alunos não será lido.\n\n"
        "Para ativar: configure `Google01` nos Secrets do Streamlit Cloud."
    )

st.markdown("### 1. Envie as fotos dos gabaritos")
uploaded = st.file_uploader(
    "Selecione uma ou várias fotos (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded:
    st.markdown(f"**{len(uploaded)} foto(s) carregada(s)**")

    if st.button("▶ Processar gabaritos", type="primary"):
        resultados = []
        progresso = st.progress(0, text="Iniciando...")
        status    = st.empty()

        for i, arq in enumerate(uploaded):
            status.text(f"Processando {arq.name} ({i+1}/{len(uploaded)})...")
            arq.seek(0)
            r = processar_arquivo(arq)
            resultados.append(r)
            progresso.progress((i + 1) / len(uploaded),
                               text=f"{i+1}/{len(uploaded)} processados")

        progresso.empty()
        status.empty()

        # ── Tabela de resultados ──────────────────────────────────────────────
        st.markdown("### 2. Resultado")

        obj_q  = ["3", "4", "5", "6", "7", "8"]
        soma_q = ["9", "10"]

        table_data = []
        for r in resultados:
            row = {"Arquivo": r["arquivo"], "Nome": r["nome"]}
            for q in obj_q:  row[f"Q{q}"]     = r["obj"].get(q, "—")
            for q in soma_q: row[f"Soma {q}"]  = r["soma"].get(q, "—")
            if r["erro"]:    row["Erro"] = r["erro"]
            table_data.append(row)

        st.dataframe(table_data, use_container_width=True)

        # ── Download CSV ──────────────────────────────────────────────────────
        csv_str  = gerar_csv(resultados)
        agora    = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resultado_turma_{agora}.csv"

        st.download_button(
            label="⬇ Baixar planilha CSV",
            data=csv_str.encode("utf-8"),
            file_name=filename,
            mime="text/csv",
        )

        erros = [r for r in resultados if r["erro"]]
        if erros:
            st.warning(f"{len(erros)} foto(s) com erro:")
            for r in erros:
                st.text(f"  {r['arquivo']}: {r['erro']}")
        else:
            st.success("Todos os gabaritos foram lidos com sucesso!")

st.divider()
st.caption("CCF · Leitor de Gabaritos · OMR + Gemini")
