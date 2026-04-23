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
import fitz  # pymupdf

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
[data-testid="block-container"] { padding-top: 0 !important; padding-bottom: 1rem !important; }
.block-container { padding-top: 0.5rem !important; padding-left: 1.5rem !important; padding-right: 1.5rem !important; }

/* Reduz espaçamento entre elementos */
[data-testid="stVerticalBlock"] { gap: 0.3rem !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; }

/* Reduz margem dos widgets */
div[data-testid="stSelectbox"],
div[data-testid="stTextInput"],
div[data-testid="stNumberInput"],
div[data-testid="stFileUploader"] { margin-bottom: 0.2rem !important; }

/* Reduz padding dos labels mas mantém visíveis */
label { margin-bottom: 0 !important; padding-bottom: 0 !important; font-size: 12px !important; font-weight: 600 !important; color: #7B1518 !important; }

/* Reduz espaçamento do st.info e st.caption */
[data-testid="stCaptionContainer"] { margin-top: 0 !important; margin-bottom: 0.2rem !important; }
[data-testid="stAlert"] { padding: 8px 12px !important; margin-bottom: 0.4rem !important; }

/* ── Cabeçalho ── */
.ccf-header {
    background: linear-gradient(135deg, #5C0F12, #7B1518 60%, #9B2528);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
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
    padding: 12px 14px 6px;
    margin-bottom: 8px;
    box-shadow: 0 1px 5px rgba(123,21,24,0.05);
}
.ccf-card-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #C4982A;
    margin-bottom: 6px;
}
.ccf-section-bar {
    background: #7B1518;
    color: white;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .5px;
    margin: 6px 0 4px;
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
    background: white !important;
    color: #2C0A0B !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}
div[data-baseweb="input"] input:focus { border-color: #C4982A !important; }
div[data-baseweb="input"] input::placeholder { color: #B0A090 !important; font-weight: 400 !important; }

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


def expandir_pdf(uploaded_file) -> list:
    """Converte cada página de um PDF em um arquivo temporário JPG.
    Retorna lista de (nome_exibido, caminho_tmp)."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    paginas = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(2.0, 2.0)   # 2× zoom → ~150 dpi → boa qualidade para OMR
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.write(pix.tobytes("jpeg", jpg_quality=95))
        tmp.close()
        nome = f"{uploaded_file.name} — pág. {i + 1}"
        paginas.append((nome, tmp.name))
    doc.close()
    return paginas


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


# ── Lógica UFSC para somatórias ──────────────────────────────────────────────
PROP_VALUES = [1, 2, 4, 8, 16, 32, 64]

def _decode_props(value: int, np: int) -> set:
    """Decodifica um valor inteiro nas proposições marcadas (potências de 2)."""
    vals = PROP_VALUES[:np]
    marked = set()
    rem = value
    for i in range(np - 1, -1, -1):
        if rem >= vals[i]:
            rem -= vals[i]
            marked.add(i)
    return marked  # rem != 0 significa valor inválido para esse np, retorna parcial

def _score_ufsc(np: int, gab_val: int, stu_val: int) -> float:
    """Pontuação UFSC: (np - erros) / np se acertos > erros, senão 0."""
    gab = _decode_props(gab_val, np)
    stu = _decode_props(stu_val, np)
    ntpc = len(gab)
    npc  = len(gab & stu)   # proposições certas do aluno
    npi  = len(stu - gab)   # proposições erradas do aluno
    if ntpc > 0 and npc > npi:
        return max(0.0, min(1.0, (np - (ntpc - (npc - npi))) / np))
    return 0.0


def calcular_nota(r: dict, gabarito: dict, peso_soma: float) -> dict:
    obj_q = ["3", "4", "5", "6", "7", "8"]
    acertos_obj = 0
    for q in obj_q:
        if gabarito.get(f"Q{q}") and r["obj"].get(q) == gabarito[f"Q{q}"]:
            acertos_obj += 1

    scores_soma = {}
    for q in ["9", "10"]:
        correta  = gabarito.get(f"SOMA{q}")
        np_count = gabarito.get(f"NP{q}", 4)
        resposta = r["soma"].get(q)
        if correta is None:
            scores_soma[q] = None
        elif resposta is not None:
            scores_soma[q] = _score_ufsc(np_count, int(correta), int(resposta))
        else:
            scores_soma[q] = 0.0

    nota_obj  = float(acertos_obj)
    nota_soma = sum(s * peso_soma for s in scores_soma.values() if s is not None)
    return {
        "acertos_obj":  acertos_obj,
        "scores_soma":  scores_soma,   # float 0.0–1.0 por questão
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
        headers += ["Acertos_Obj", "Score_Soma", "Nota_Total"]
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
            scores = calc["scores_soma"]
            soma_str = " | ".join(
                f"Q{q}:{scores[q]*100:.0f}%" for q in ["9","10"] if scores.get(q) is not None
            )
            row["Acertos_Obj"]   = calc["acertos_obj"]
            row["Score_Soma"]    = soma_str
            row["Nota_Total"]    = f"{calc['nota_total']:.2f}"
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
st.info("✏️ **Como preencher:** Digite a letra da resposta correta em **CAIXA ALTA** (A, B, C, D ou E). Deixe o campo em branco se não quiser corrigir aquela questão.")

cols_obj = st.columns(6)
for i, q in enumerate(["3", "4", "5", "6", "7", "8"]):
    with cols_obj[i]:
        val = st.text_input(f"Q{int(q):02d}", placeholder="A–E", max_chars=1, key=f"gab_q{q}")
        val = val.strip().upper()
        gabarito[f"Q{q}"] = val if val in ("A", "B", "C", "D", "E") else ""

st.markdown('<div class="ccf-section-bar">Questões Somatórias — Q09 e Q10</div>', unsafe_allow_html=True)
st.info(
    "✏️ **Como preencher:** Digite o **valor do gabarito** (soma das proposições verdadeiras, de 0 a 99). "
    "Selecione também o **número de proposições** da questão (4, 5, 6 ou 7). "
    "A pontuação é calculada pelo método UFSC: acertos parciais são considerados. "
    "Deixe em branco para não corrigir."
)

def _max_soma(np: int) -> int:
    return sum(PROP_VALUES[:np])

def _valida_soma(valor_str: str, np: int, label: str):
    """Exibe aviso se o valor digitado for inválido para o NP escolhido."""
    if not valor_str.strip():
        return
    try:
        v = int(valor_str)
    except ValueError:
        st.error(f"{label}: valor deve ser um número inteiro.")
        return
    maximo = _max_soma(np)
    if v < 0 or v > maximo:
        st.error(
            f"{label}: valor **{v}** inválido para NP={np}. "
            f"Com {np} proposições o máximo é **{maximo}** "
            f"({' + '.join(str(PROP_VALUES[i]) for i in range(np))})."
        )
    else:
        props = _decode_props(v, np)
        if props:
            nomes = " + ".join(str(PROP_VALUES[i]) for i in sorted(props))
            st.caption(f"Proposições verdadeiras: {nomes}")
        else:
            st.caption("Nenhuma proposição verdadeira (valor 0).")

col_s1, col_s2 = st.columns([1, 1])
with col_s1:
    np9 = st.selectbox("Q09 — Nº de proposições", [4, 5, 6, 7], key="np9")
    gabarito["NP9"] = np9
    vals9 = ", ".join(str(PROP_VALUES[i]) for i in range(np9))
    v9_str = st.text_input(f"Q09 — Valor do gabarito (máx {_max_soma(np9)})", help=f"Proposições disponíveis: {vals9}", placeholder="Ex: 13", key="gab_soma9")
    _valida_soma(v9_str, np9, "Q09")
    try:
        gabarito["SOMA9"] = int(v9_str) if v9_str.strip() else None
    except ValueError:
        gabarito["SOMA9"] = None

with col_s2:
    np10 = st.selectbox("Q10 — Nº de proposições", [4, 5, 6, 7], key="np10")
    gabarito["NP10"] = np10
    vals10 = ", ".join(str(PROP_VALUES[i]) for i in range(np10))
    v10_str = st.text_input(f"Q10 — Valor do gabarito (máx {_max_soma(np10)})", help=f"Proposições disponíveis: {vals10}", placeholder="Ex: 06", key="gab_soma10")
    _valida_soma(v10_str, np10, "Q10")
    try:
        gabarito["SOMA10"] = int(v10_str) if v10_str.strip() else None
    except ValueError:
        gabarito["SOMA10"] = None

peso_soma = 1.0

st.markdown('</div>', unsafe_allow_html=True)

# ── PASSO 2: Upload ───────────────────────────────────────────────────────────
st.markdown('<div class="ccf-card"><div class="ccf-card-label">Passo 2 — Fotos dos gabaritos</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Arraste as fotos ou PDFs aqui (JPG, PNG, WEBP, PDF)",
    type=["jpg", "jpeg", "png", "webp", "pdf"],
    accept_multiple_files=True,
)
if uploaded:
    n_pdf  = sum(1 for f in uploaded if f.name.lower().endswith(".pdf"))
    n_img  = len(uploaded) - n_pdf
    partes = []
    if n_img:  partes.append(f"{n_img} imagem(ns)")
    if n_pdf:  partes.append(f"{n_pdf} PDF(s)")
    st.caption(f"{' e '.join(partes)} selecionado(s) — cada página de PDF será processada individualmente")
st.markdown('</div>', unsafe_allow_html=True)

# ── Botão processar ───────────────────────────────────────────────────────────
if uploaded:
    if st.button("▶  Processar gabaritos", type="primary", use_container_width=True):
        resultados = []
        progresso  = st.progress(0, text="Iniciando...")
        status     = st.empty()

        # Expande PDFs em páginas individuais
        fila = []   # lista de (nome_exibido, path_tmp | UploadedFile)
        tmps_pdf = []
        for arq in uploaded:
            if arq.name.lower().endswith(".pdf"):
                arq.seek(0)
                paginas = expandir_pdf(arq)
                fila.extend(paginas)
                tmps_pdf.extend(p for _, p in paginas)
            else:
                fila.append((arq.name, arq))

        total = len(fila)
        for i, (nome_exib, origem) in enumerate(fila):
            status.text(f"Lendo {nome_exib} ({i+1}/{total})...")
            if isinstance(origem, str):
                # página de PDF: já é um path temporário
                try:
                    resultado = process_answer_sheet(origem)
                    if "erro" in resultado:
                        r = {"arquivo": nome_exib, "erro": resultado["erro"],
                             "nome": "", "obj": {}, "soma": {}}
                    else:
                        obj_raw  = resultado.get("objetivas", {})
                        soma_raw = resultado.get("somatorias", {})
                        nome_al  = ler_nome_via_gemini(origem)
                        r = {
                            "arquivo": nome_exib,
                            "nome":    nome_al,
                            "obj":     {str(k): v for k, v in obj_raw.items()  if v is not None},
                            "soma":    {str(k): v for k, v in soma_raw.items() if v is not None},
                            "erro":    "",
                        }
                except Exception as e:
                    r = {"arquivo": nome_exib, "erro": str(e),
                         "nome": "", "obj": {}, "soma": {}}
            else:
                origem.seek(0)
                r = processar_arquivo(origem)
                r["arquivo"] = nome_exib
            resultados.append(r)
            progresso.progress((i + 1) / total,
                               text=f"{i+1} de {total} processados")

        for p in tmps_pdf:
            try: os.unlink(p)
            except: pass

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
                val = r["soma"].get(q, "")
                if tem_gabarito and not r["erro"] and gabarito.get(f"SOMA{q}") is not None and val != "":
                    np_q  = gabarito.get(f"NP{q}", 4)
                    maximo_q = _max_soma(np_q)
                    val_int = int(val)
                    if val_int > maximo_q:
                        row[f"Soma {q}"] = f"⚠️ {val} (inválido — máx {maximo_q} para NP={np_q})"
                    else:
                        score = _score_ufsc(np_q, int(gabarito[f"SOMA{q}"]), val_int)
                        pct   = int(score * 100)
                        icon  = "✅" if pct == 100 else ("⚠️" if pct > 0 else "❌")
                        row[f"Soma {q}"] = f"{icon} {val} ({pct}%)"
                else:
                    row[f"Soma {q}"] = str(val) if val != "" else "—"

            if tem_gabarito and not r["erro"]:
                calc = calcular_nota(r, gabarito, peso_soma)
                row["Nota"] = f"{calc['nota_total']:.2f}"

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
