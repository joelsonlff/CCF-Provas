import os
import sys
import glob
import json
import time
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

# ── Modo de processamento ─────────────────────────────────────────────────────
# Uso: python main.py [--modo gemini|omr]
# Padrão: gemini (usa API do Google)
#         omr    (processamento local com OpenCV + Gemini só para ler o nome)

MODO = "gemini"
if "--modo" in sys.argv:
    idx = sys.argv.index("--modo")
    if idx + 1 < len(sys.argv):
        MODO = sys.argv[idx + 1].lower()

if MODO not in ("gemini", "omr"):
    print(f"ERRO: modo inválido '{MODO}'. Use --modo gemini ou --modo omr")
    sys.exit(1)

if MODO == "omr":
    try:
        from services.omr_processor import process_answer_sheet
    except ImportError as e:
        print(f"ERRO: Módulo OMR não encontrado. Execute: pip install opencv-python numpy")
        print(f"Detalhe: {e}")
        sys.exit(1)

# ── Configuração Gemini ───────────────────────────────────────────────────────
load_dotenv()

api_keys = []
for k, v in os.environ.items():
    if k.lower().startswith('google') and v.strip():
        api_keys.append(v.strip())
if not api_keys:
    k = os.getenv("GEMINI_API_KEY")
    if k:
        api_keys.append(k)

if not api_keys:
    if MODO == "gemini":
        print("-> ERRO CRÍTICO: Nenhuma chave API configurada no .env.")
        print("   Copie o arquivo .env.example para .env e coloque suas chaves lá.")
        print("   Ou use o modo local: python main.py --modo omr")
        sys.exit(1)
    else:
        print("[AVISO] Nenhuma chave API configurada — nome do aluno não será lido.")

# Rodízio de chaves thread-safe
_key_lock = threading.Lock()
current_key_idx = 0

def get_current_key():
    with _key_lock:
        return api_keys[current_key_idx]

def next_key():
    global current_key_idx
    with _key_lock:
        current_key_idx = (current_key_idx + 1) % len(api_keys)
        print(f"*** Alternando limite da API. Usando chave: {current_key_idx+1} de {len(api_keys)} ***")

def safe_gemini_call(img, prompt, json_mode=True):
    if not api_keys:
        return '{"erro": "Sem chave API configurada"}'
    max_retries = len(api_keys) * 2
    retries = 0
    while retries < max_retries:
        try:
            api_key = get_current_key()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            kwargs = {"temperature": 0.0}
            if json_mode:
                kwargs["response_mime_type"] = "application/json"
            response = model.generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(**kwargs)
            )
            return response.text
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "exhausted" in err_str or "503" in err_str:
                next_key()
                time.sleep(1)
                retries += 1
            else:
                return f'{{"erro": "Falha na API: {str(e)}"}}'

    return '{"erro": "Limite atingido em TODAS as chaves configuradas. Tente novamente mais tarde."}'

def resize_image(image_path, max_size=1600):
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    w, h = img.size
    if w > max_size or h > max_size:
        if w > h:
            new_h = int(h * max_size / w)
            new_w = max_size
        else:
            new_w = int(w * max_size / h)
            new_h = max_size
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img

def ler_nome_via_gemini(file_path):
    """Chamada mínima ao Gemini apenas para ler o nome do aluno no cabeçalho."""
    if not api_keys:
        return ""
    try:
        img = resize_image(file_path, max_size=800)
        prompt = (
            "Esta é uma folha de resposta escolar. No cabeçalho há um campo 'ALUNO' "
            "preenchido pelo aluno com letra de forma em caixa alta. "
            "Leia o nome escrito nesse campo e responda APENAS com o nome, "
            "sem pontuação, sem explicação, sem aspas. "
            "Se não conseguir ler, responda apenas: (ilegível)"
        )
        nome = safe_gemini_call(img, prompt, json_mode=False).strip()
        # Remove aspas ou lixo que o modelo eventualmente adiciona
        nome = nome.strip('"\'').strip()
        return nome
    except Exception:
        return ""

def build_prompt():
    nAlts = 5
    OPTS = ["A", "B", "C", "D", "E"]
    return (
        'Voce eh um sistema especializado em leitura optica de gabaritos escolares. Analise a imagem com MAXIMA ATENCAO.'
        ' ESTRUTURA DA FOLHA (de cima para baixo):'
        ' [CABECALHO] Logo CCF + campo ALUNO escrito pelo aluno em caixa alta + QR Codes de referencia.'
        ' [Q01 e Q02] Blocos com linhas para escrita — SAO DISCURSIVAS, ignore.'
        f' [QUESTOES OBJETIVAS] 6 questoes (Q03 a Q08) em grade 2 colunas. Cada questao tem {nAlts} BOLHAS CIRCULARES grandes com letras {" ".join(OPTS)} da esquerda para direita. O aluno PINTA/PREENCHE completamente uma bolha com caneta azul ou preta. A bolha pintada fica ESCURA E PREENCHIDA, as demais ficam VAZIAS com apenas o contorno e a letra.'
        ' [QUESTOES SOMATORIAS] 2 questoes (Q09 e Q10). Cada questao tem DUAS FILEIRAS de 10 bolhas numeradas 0 1 2 3 4 5 6 7 8 9 da esquerda para direita. Fileira superior tem rotulo "Dez:" e fileira inferior tem rotulo "Uni:". O aluno pinta UMA bolha em cada fileira. Leia o NUMERO dentro da bolha pintada em cada fileira. Soma = Dez*10 + Uni.'
        ' COMO IDENTIFICAR BOLHA PINTADA: ela aparece PREENCHIDA/SOLIDA/ESCURA em contraste com as bolhas vizinhas que sao VAZIAS (so tem contorno e letra/numero). Em fotos pode aparecer azul escuro ou roxo escuro (cor da tinta da caneta).'
        f' PROCESSO: 1) Leia nome do aluno. 2) Para cada questao objetiva, olhe as {nAlts} bolhas e identifique qual esta preenchida. 3) Para cada somatoria, identifique a bolha preenchida na fileira Dez e na fileira Uni.'
        ' REGRA ABSOLUTA: sua resposta deve conter APENAS o JSON abaixo, sem nenhum texto antes ou depois, sem blocos de código, sem explicações:'
        ' {"nome":"NOME","obj":{"3":"A","4":"B","5":"C","6":"D","7":"E","8":"A"},"soma":{"9":20,"10":10},'
        ' "qr":{"prof":"","disc":"","turma":"","trim":"","aval":""},'
        ' "obs":"Q03:A(pintada) Q04:E(pintada) Q05:C(pintada) etc — descreva cada questao"}'
        ' Se objetiva estiver em branco use null. soma deve ser inteiro 0-99. Se imagem ilegivel: {"erro":"motivo"}.'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE PROCESSAMENTO
# ═══════════════════════════════════════════════════════════════════════════════

def processar_imagem_gemini(file_path, prompt, total, idx):
    """Processa uma imagem via Gemini API. Retorna dados no formato interno."""
    nome_arq = os.path.basename(file_path)
    print(f" -> Enviando [{idx}/{total}]: {nome_arq} ...", flush=True)
    try:
        img = resize_image(file_path, max_size=1600)
        resposta_json_str = safe_gemini_call(img, prompt)

        if '```json' in resposta_json_str:
            resposta_json_str = resposta_json_str.split('```json')[1]
            if '```' in resposta_json_str:
                resposta_json_str = resposta_json_str.rsplit('```', 1)[0]
        elif '```' in resposta_json_str:
            resposta_json_str = resposta_json_str.replace('```', '')

        resposta_json_str = resposta_json_str.strip()

        try:
            dados = json.loads(resposta_json_str)
        except json.JSONDecodeError as e:
            print(f" Erro JSON [{nome_arq}]: {e}")
            dados = {"erro": "Resposta fora do padrao JSON pela IA."}

        dados['arquivo'] = nome_arq

        if 'erro' not in dados:
            print(f" OK [{nome_arq}] | Aluno: {dados.get('nome', '?')}")
        else:
            print(f" Falha [{nome_arq}]: {dados.get('erro', '')}")

        return dados

    except Exception as e:
        print(f" Erro interno [{nome_arq}]: {e}")
        return {"arquivo": nome_arq, "erro": str(e)}


def processar_imagem_omr(file_path, total, idx):
    """
    Processa uma imagem via OMR local (OpenCV) para as bolhas +
    Gemini para ler o nome do aluno no cabeçalho.
    """
    nome_arq = os.path.basename(file_path)
    print(f" -> OMR [{idx}/{total}]: {nome_arq} ...", flush=True)
    try:
        resultado = process_answer_sheet(file_path)

        if "erro" in resultado:
            print(f" Falha [{nome_arq}]: {resultado['erro']}")
            return {"arquivo": nome_arq, "erro": resultado["erro"]}

        obj_raw = resultado.get("objetivas", {})
        soma_raw = resultado.get("somatorias", {})

        # Lê o nome via Gemini (chamada mínima, imagem pequena)
        nome_aluno = ler_nome_via_gemini(file_path)

        dados = {
            "arquivo": nome_arq,
            "nome": nome_aluno,
            "obj": {str(k): v for k, v in obj_raw.items() if v is not None},
            "soma": {str(k): v for k, v in soma_raw.items() if v is not None},
            "obs": f"[OMR local] Folha detectada: {resultado.get('sheet_detected', False)}",
        }

        obj_str = " ".join(f"Q{k}:{v}" for k, v in sorted(obj_raw.items()) if v)
        soma_str = " ".join(f"Q{k}:{v}" for k, v in sorted(soma_raw.items()) if v is not None)
        print(f" OK [{nome_arq}] | Aluno: {nome_aluno or '(sem nome)'} | {obj_str} | {soma_str}")

        return dados

    except Exception as e:
        print(f" Erro interno [{nome_arq}]: {e}")
        return {"arquivo": nome_arq, "erro": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    pasta_img = "imagens_provas"
    pasta_res = "resultados"

    if not os.path.exists(pasta_img): os.makedirs(pasta_img)
    if not os.path.exists(pasta_res): os.makedirs(pasta_res)

    arquivos = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.webp'):
        arquivos.extend(glob.glob(os.path.join(pasta_img, ext)))
        arquivos.extend(glob.glob(os.path.join(pasta_img, ext.upper())))
    arquivos = list(set(arquivos))

    if not arquivos:
        print(f"\n[Ação Necessária] Nenhuma imagem encontrada em '{pasta_img}'.")
        print("Coloque as fotos nessa pasta e tente novamente.")
        return

    total = len(arquivos)

    if MODO == "gemini":
        n_workers = min(len(api_keys), 5)
        print("Iniciando leitor de gabaritos — modo: Gemini API")
        print(f"Detectadas {len(api_keys)} chaves | {n_workers} workers paralelos | {total} imagens")

        prompt = build_prompt()
        resultados_map = {}

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futuros = {
                executor.submit(processar_imagem_gemini, fp, prompt, total, i + 1): fp
                for i, fp in enumerate(arquivos)
            }
            for futuro in as_completed(futuros):
                file_path = futuros[futuro]
                try:
                    dados = futuro.result()
                except Exception as e:
                    dados = {"arquivo": os.path.basename(file_path), "erro": str(e)}
                resultados_map[file_path] = dados

    else:  # MODO == "omr"
        n_workers = min(os.cpu_count() or 4, 8)
        tem_api = bool(api_keys)
        print("Iniciando leitor de gabaritos — modo: OMR local (OpenCV)")
        print(f"Leitura de nome via Gemini: {'sim' if tem_api else 'não (sem chave API)'}")
        print(f"{n_workers} workers | {total} imagens")

        resultados_map = {}

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futuros = {
                executor.submit(processar_imagem_omr, fp, total, i + 1): fp
                for i, fp in enumerate(arquivos)
            }
            for futuro in as_completed(futuros):
                file_path = futuros[futuro]
                try:
                    dados = futuro.result()
                except Exception as e:
                    dados = {"arquivo": os.path.basename(file_path), "erro": str(e)}
                resultados_map[file_path] = dados

    # Preserva ordem original
    resultados = [resultados_map[fp] for fp in arquivos]

    # ── Gerar CSV ─────────────────────────────────────────────────────────────
    agora = datetime.now().strftime("%Y%m%d_%H%M%S")
    sufixo = "omr" if MODO == "omr" else "gemini"
    csv_path = os.path.join(pasta_res, f"resultado_turma_{sufixo}_{agora}.csv")

    obj_q  = ['3', '4', '5', '6', '7', '8']
    soma_q = ['9', '10']
    headers = ['Arquivo', 'NomeAluno'] + [f'Q{q}' for q in obj_q] + [f'SOMA_{q}' for q in soma_q] + ['Obs', 'Erro']

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=';')
        writer.writeheader()
        for r in resultados:
            obj  = r.get('obj', {})
            soma = r.get('soma', {})
            row = {
                'Arquivo':   r.get('arquivo', ''),
                'NomeAluno': r.get('nome', ''),
                'Obs':       r.get('obs', ''),
                'Erro':      r.get('erro', ''),
            }
            for q in obj_q:  row[f'Q{q}']    = obj.get(q, '')
            for q in soma_q: row[f'SOMA_{q}'] = soma.get(q, '')
            writer.writerow(row)

    print(f"\nFinalizado! Planilha salva em: {csv_path}")


if __name__ == "__main__":
    main()
