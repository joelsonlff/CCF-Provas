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
import base64
import threading
from pathlib import Path
from datetime import datetime

import google.generativeai as genai
from PIL import Image

from services.omr_processor import process_answer_sheet

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Leitor de Gabaritos — CCF",
    page_icon="📝",
    layout="centered",
)

# ── Logo em base64 ────────────────────────────────────────────────────────────
_logo_path = Path(__file__).parent / "assets" / "logo_ccf.png"
_logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode() if _logo_path.exists() else ""

# ── Estilo visual (tema CCF) ──────────────────────────────────────────────────
st.markdown("""
<style>
/* Remove padding padrão do Streamlit */
[data-testid="stAppViewContainer"] { background: #FDF8F0; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="block-container"] { padding-top: 0 !important; }
.block-container { padding-top: 1rem !important; }

/* ── Cabeçalho ── */
.ccf-header {
    background: linear-gradient(135deg, #5C0F12, #7B1518 60%, #9B2528);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    gap: 16px;
}
.ccf-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: #C4982A;
}
.ccf-logo-box {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(196,152,42,0.35);
    border-radius: 10px;
    padding: 6px 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
}
.ccf-logo-box img { height: 42px; width: auto; }
.ccf-logo-text strong { font-size: 11px; font-weight: 700; color: white; display: block; }
.ccf-logo-text span  { font-size: 9px; color: rgba(255,255,255,0.6); }
.ccf-title { flex: 1; }
.ccf-title h1 { font-size: 18px; font-weight: 700; color: white; margin: 0 0 2px; }
.ccf-title p  { font-size: 11px; color: rgba(255,255,255,0.6); margin: 0; }
.ccf-badge {
    background: rgba(30,138,76,0.25);
    border: 1px solid rgba(30,138,76,0.5);
    color: #7fffc0;
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 20px;
    white-space: nowrap;
    flex-shrink: 0;
}

/* ── Cards ── */
.ccf-card {
    background: white;
    border: 1px solid #E0D5C5;
    border-radius: 12px;
    padding: 18px 18px 8px;
    margin-bottom: 14px;
    box-shadow: 0 1px 5px rgba(123,21,24,0.05);
}
.ccf-card-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #C4982A;
    margin-bottom: 10px;
}
.ccf-section-bar {
    background: #7B1518;
    color: white;
    padding: 6px 12px;
    border-radius: 7px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .5px;
    margin: 10px 0 8px;
}

/* ── Botões ── */
div.stButton > button[kind="primary"] {
    background: #7B1518 !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 9px !important;
    padding: 10px 20px !important;
    font-size: 14px !important;
}
div.stButton > button[kind="primary"]:hover { background: #5C0F12 !important; }

div.stDownloadButton > button {
    background: white !important;
    border: 1.5px solid #7B1518 !important;
    color: #7B1518 !important;
    font-weight: 700 !important;
    border-radius: 9px !important;
}
div.stDownloadButton > button:hover { background: #FDF8F0 !important; }

/* ── Radio buttons do gabarito ── */
div[data-testid="stRadio"] label { font-size: 13px !important; }
div[data-testid="stRadio"] > div { gap: 4px !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #E0D5C5 !important;
    border-radius: 12px !important;
    background: #F5F0E8 !important;
    padding: 10px !important;
}

/* ── Inputs ── */
div[data-baseweb="input"] input {
    border-color: #E0D5C5 !important;
    background: #FDF8F0 !important;
}
div[data-baseweb="input"] input:focus { border-color: #C4982A !important; }

/* ── Rodapé ── */
.ccf-footer {
    text-align: center;
    font-size: 11px;
    color: #8B5E5F;
    margin-top: 20px;
    padding-top: 14px;
    border-top: 1px solid #E0D5C5;
}
</style>
""", unsafe_allow_html=True)

# ── Chave API ─────────────────────────────────────────────────────────────────
def get_api_keys():
    keys = []
    try:
        for k, v in st.secrets.items():
            if k.lower().startswith("google") and str(v).strip():
                keys.append(str(v).strip())
        if not keys and "GEMINI_API_KEY" in st.secrets:
            keys.append(str(st.secrets["GEMINI_API_KEY"]).strip())
    except Exception:
        pass
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


def calcular_nota(r: dict, gabarito: dict, peso_soma: float) -> dict:
    obj_q = ["3", "4", "5", "6", "7", "8"]
    acertos_obj = 0
    for q in obj_q:
        if gabarito.get(f"Q{q}") and r["obj"].get(q) == gabarito[f"Q{q}"]:
            acertos_obj += 1

    acertos_soma = {}
    for q in ["9", "10"]:
        correta = gabarito.get(f"SOMA{q}")
        resposta = r["soma"].get(q)
        if correta is None:
            acertos_soma[q] = None
        elif resposta is not None and int(resposta) == int(correta):
            acertos_soma[q] = True
        else:
            acertos_soma[q] = False

    nota_obj   = float(acertos_obj)
    nota_soma  = sum(peso_soma for ok in acertos_soma.values() if ok)
    return {
        "acertos_obj":  acertos_obj,
        "acertos_soma": acertos_soma,
        "nota_total":   nota_obj + nota_soma,
    }


def gerar_csv(resultados: list, gabarito: dict, peso_soma: float) -> str:
    obj_q  = ["3", "4", "5", "6", "7", "8"]
    soma_q = ["9", "10"]
    tem_gabarito = any(v is not None and v != "" for v in gabarito.values())

    headers = ["Arquivo", "NomeAluno"]
    headers += [f"Q{q}" for q in obj_q]
    headers += [f"SOMA_{q}" for q in soma_q]
    if tem_gabarito:
        headers += ["Acertos_Obj", "Acertos_Soma", "Nota_Total"]
    headers += ["Erro"]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, delimiter=";")
    writer.writeheader()
    for r in resultados:
        row = {"Arquivo": r["arquivo"], "NomeAluno": r["nome"], "Erro": r["erro"]}
        for q in obj_q:  row[f"Q{q}"]    = r["obj"].get(q, "")
        for q in soma_q: row[f"SOMA_{q}"] = r["soma"].get(q, "")
        if tem_gabarito and not r["erro"]:
            calc = calcular_nota(r, gabarito, peso_soma)
            soma_acertos = sum(1 for ok in calc["acertos_soma"].values() if ok)
            row["Acertos_Obj"]  = calc["acertos_obj"]
            row["Acertos_Soma"] = soma_acertos
            row["Nota_Total"]   = f"{calc['nota_total']:.1f}"
        writer.writerow(row)
    return buf.getvalue()


# ════════════════════════════════════════════════════════
# INTERFACE
# ════════════════════════════════════════════════════════

# ── Cabeçalho com logo ────────────────────────────────────────────────────────
logo_img_tag = (
    f'<img src="data:image/png;base64,{_logo_b64}" alt="CCF">'
    if _logo_b64 else "📝"
)
st.markdown(f"""
<div class="ccf-header">
  <div class="ccf-logo-box">
    {logo_img_tag}
    <div class="ccf-logo-text">
      <strong>COLÉGIO CORAÇÃO FELIZ</strong>
      <span>Tubarão — SC</span>
    </div>
  </div>
  <div class="ccf-title">
    <h1>Leitor de Gabaritos</h1>
    <p>Correção automática de folhas de resposta</p>
  </div>
  <span class="ccf-badge">OMR + Gemini AI</span>
</div>
""", unsafe_allow_html=True)

if not API_KEYS:
    st.warning("Nenhuma chave API configurada — o nome dos alunos não será lido. Configure `Google01` nos Secrets do Streamlit Cloud.")

# ── PASSO 1: Gabarito ─────────────────────────────────────────────────────────
st.markdown('<div class="ccf-card"><div class="ccf-card-label">Passo 1 — Gabarito (opcional)</div>', unsafe_allow_html=True)
st.caption("Preencha para calcular nota automaticamente. Deixe em branco para ver apenas as respostas.")

gabarito = {}

st.markdown('<div class="ccf-section-bar">Questões Objetivas — Q03 a Q08</div>', unsafe_allow_html=True)
st.caption("Digite a letra correta em CAIXA ALTA (A, B, C, D ou E). Deixe em branco para não corrigir.")

cols_obj = st.columns(6)
for i, q in enumerate(["3", "4", "5", "6", "7", "8"]):
    with cols_obj[i]:
        val = st.text_input(f"Q{q}", placeholder="A–E", max_chars=1, key=f"gab_q{q}")
        val = val.strip().upper()
        gabarito[f"Q{q}"] = val if val in ("A", "B", "C", "D", "E") else ""

st.markdown('<div class="ccf-section-bar">Questões Somatórias — Q09 e Q10</div>', unsafe_allow_html=True)
st.caption("Digite o valor numérico correto (0 a 99) em CAIXA ALTA. Deixe em branco para não corrigir.")

col_s1, col_s2, col_s3 = st.columns([2, 2, 2])
with col_s1:
    v9_str = st.text_input("Q09 — valor correto", placeholder="Ex: 45", key="gab_soma9")
    try:
        gabarito["SOMA9"] = int(v9_str) if v9_str.strip() else None
    except ValueError:
        gabarito["SOMA9"] = None
        st.caption("⚠️ Digite apenas números")

with col_s2:
    v10_str = st.text_input("Q10 — valor correto", placeholder="Ex: 30", key="gab_soma10")
    try:
        gabarito["SOMA10"] = int(v10_str) if v10_str.strip() else None
    except ValueError:
        gabarito["SOMA10"] = None
        st.caption("⚠️ Digite apenas números")

with col_s3:
    peso_soma = st.number_input(
        "Pontos por somatória correta",
        min_value=0.0, max_value=10.0, value=1.0, step=0.5,
        key="peso_soma"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── PASSO 2: Upload ───────────────────────────────────────────────────────────
st.markdown('<div class="ccf-card"><div class="ccf-card-label">Passo 2 — Fotos dos gabaritos</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Arraste as fotos aqui ou clique para selecionar (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)
if uploaded:
    st.caption(f"{len(uploaded)} foto(s) selecionada(s)")
st.markdown('</div>', unsafe_allow_html=True)

# ── Botão processar ───────────────────────────────────────────────────────────
if uploaded:
    if st.button("▶  Processar gabaritos", type="primary", use_container_width=True):
        resultados = []
        progresso  = st.progress(0, text="Iniciando...")
        status     = st.empty()

        for i, arq in enumerate(uploaded):
            status.text(f"Lendo {arq.name} ({i+1}/{len(uploaded)})...")
            arq.seek(0)
            r = processar_arquivo(arq)
            resultados.append(r)
            progresso.progress((i + 1) / len(uploaded),
                               text=f"{i+1} de {len(uploaded)} processados")

        progresso.empty()
        status.empty()

        # ── PASSO 3: Resultado ────────────────────────────────────────────────
        st.markdown('<div class="ccf-card"><div class="ccf-card-label">Passo 3 — Resultado</div>', unsafe_allow_html=True)

        obj_q  = ["3", "4", "5", "6", "7", "8"]
        soma_q = ["9", "10"]
        tem_gabarito = any(v is not None and v != "" for v in gabarito.values())

        table_data = []
        for r in resultados:
            row = {"Nome": r["nome"] or r["arquivo"]}
            for q in obj_q:
                resposta = r["obj"].get(q, "")
                correta  = gabarito.get(f"Q{q}", "") if tem_gabarito else ""
                if correta and resposta == correta:
                    row[f"Q{q}"] = f"✅ {resposta}"
                elif correta and resposta:
                    row[f"Q{q}"] = f"❌ {resposta}"
                else:
                    row[f"Q{q}"] = resposta or "—"

            for q in soma_q:
                val     = r["soma"].get(q, "")
                correta = gabarito.get(f"SOMA{q}") if tem_gabarito else None
                if correta is not None and val != "":
                    row[f"Soma {q}"] = f"✅ {val}" if int(val) == int(correta) else f"❌ {val}"
                else:
                    row[f"Soma {q}"] = str(val) if val != "" else "—"

            if tem_gabarito and not r["erro"]:
                calc = calcular_nota(r, gabarito, peso_soma)
                row["Nota"] = f"{calc['nota_total']:.1f}"

            if r["erro"]:
                row["Erro"] = r["erro"]

            table_data.append(row)

        st.dataframe(table_data, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Download ──────────────────────────────────────────────────────────
        csv_str  = gerar_csv(resultados, gabarito, peso_soma)
        agora    = datetime.now().strftime("%Y%m%d_%H%M%S")

        st.download_button(
            label="⬇  Baixar planilha CSV",
            data=csv_str.encode("utf-8"),
            file_name=f"resultado_turma_{agora}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        erros = [r for r in resultados if r["erro"]]
        if erros:
            st.warning(f"{len(erros)} foto(s) com erro de leitura:")
            for r in erros:
                st.text(f"  {r['arquivo']}: {r['erro']}")
        else:
            st.success("Todos os gabaritos foram lidos com sucesso!")

# ── Rodapé ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="ccf-footer">Colégio Coração Feliz · Tubarão — SC · Leitor de Gabaritos</div>',
    unsafe_allow_html=True
)
