import os
import glob
import json
import time
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Carrega variáveis de ambiente do .env para rodar localmente
load_dotenv()

# Pegar as chaves GoogleXX ou GEMINI_API_KEY do sistema/env
api_keys = []
for k, v in os.environ.items():
    if k.lower().startswith('google') and v.strip():
        api_keys.append(v.strip())

if not api_keys:
    k = os.getenv("GEMINI_API_KEY")
    if k:
        api_keys.append(k)
    else:
        print("-> ERRO CRÍTICO: Nenhuma chave API configurada no .env.")
        print("   Copie o arquivo .env.example para .env e coloque suas chaves lá.")
        exit(1)

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
        key = api_keys[current_key_idx]
        print(f"*** Alternando limite da API. Usando chave: {current_key_idx+1} de {len(api_keys)} ***")
    return key

def safe_gemini_call(img, prompt):
    max_retries = len(api_keys) * 2
    retries = 0
    while retries < max_retries:
        try:
            # Cada thread configura sua própria chave antes de chamar
            api_key = get_current_key()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
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
    """ Reduz a imagem pesada tirada de câmeras antes de enviar para poupar internet/tempo """
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

def processar_imagem(file_path, prompt, total, idx):
    """Processa uma única imagem e retorna o resultado. Chamada em paralelo."""
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


def main():
    # Número de workers: 1 por chave, máximo de 5 para não sobrecarregar
    n_workers = min(len(api_keys), 5)

    print("Iniciando leitor de gabaritos integrado ao sistema Python...")
    print(f"Detectadas {len(api_keys)} chaves | Processamento paralelo: {n_workers} workers")

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
        print(f"\n[Ação Necessária] Nenhuma imagem de prova foi encontrada na pasta '{pasta_img}'.")
        print("Coloque as fotos nessa pasta e tente novamente.")
        return

    total = len(arquivos)
    print(f"Foi(ram) encontrada(s) {total} imagens. Iniciando processamento em paralelo:")
    prompt = build_prompt()

    # Dicionário para manter a ordem original dos arquivos no resultado
    resultados_map = {}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futuros = {
            executor.submit(processar_imagem, fp, prompt, total, i + 1): fp
            for i, fp in enumerate(arquivos)
        }
        for futuro in as_completed(futuros):
            file_path = futuros[futuro]
            try:
                dados = futuro.result()
            except Exception as e:
                nome_arq = os.path.basename(file_path)
                dados = {"arquivo": nome_arq, "erro": str(e)}
            resultados_map[file_path] = dados

    # Preserva a ordem original dos arquivos
    resultados = [resultados_map[fp] for fp in arquivos]

    # Criar planilha CSV de resumo
    agora = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(pasta_res, f"resultado_turma_{agora}.csv")

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        obj_q = ['3','4','5','6','7','8']
        soma_q = ['9','10']

        headers = ['Arquivo', 'NomeAluno'] + [f'Q{q}' for q in obj_q] + [f'SOMA_{q}' for q in soma_q] + ['Obs', 'Erro']
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=';')
        writer.writeheader()

        for r in resultados:
            row = {'Arquivo': r.get('arquivo', ''), 'NomeAluno': r.get('nome', ''), 'Obs': r.get('obs', ''), 'Erro': r.get('erro', '')}
            obj, soma = r.get('obj', {}), r.get('soma', {})
            for q in obj_q: row[f'Q{q}'] = obj.get(q, '')
            for q in soma_q: row[f'SOMA_{q}'] = soma.get(q, '')
            writer.writerow(row)

    print(f"\nFinalizado com sucesso! Planilha guardada em: {csv_path}")

if __name__ == "__main__":
    main()
