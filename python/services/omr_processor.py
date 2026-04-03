"""
omr_processor.py
================
OMR (Optical Mark Recognition) local para folhas de resposta CCF.
Detecta bolhas preenchidas usando OpenCV — sem chamadas a APIs externas.

Layout da folha (de cima para baixo, após alinhamento):
  [Cabeçalho]  Logo + nome do aluno + QR codes
  [Q01–Q02]    Questões discursivas — ignoradas
  [Q03–Q08]    6 objetivas em grade 2 colunas × 3 linhas, 5 bolhas (A–E) cada
  [Q09–Q10]    2 somatórias, cada uma com:
                 Fileira Dez: bolhas 0–9 (uma selecionada)
                 Fileira Uni: bolhas 0–9 (uma selecionada)
                 Resultado = Dez × 10 + Uni  (inteiro 0–99)

Uso:
    from services.omr_processor import process_answer_sheet
    result = process_answer_sheet("foto.jpg")
    # {"objetivas": ["A","C","B","D","E","A"], "somatorias": [20, 10]}

Calibração:
    Execute `python services/calibrate.py foto.jpg` para visualizar a grade
    detectada e ajustar as constantes LAYOUT_* abaixo conforme necessário.
"""

import cv2
import numpy as np
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSÕES DO CANVAS APÓS WARP
# ═══════════════════════════════════════════════════════════════════════════════
WARP_W = 800   # largura do canvas normalizado (px)
WARP_H = 1100  # altura do canvas normalizado (px)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT — coordenadas normalizadas (0.0–1.0) relativas ao canvas warped
#
# ⚠  CALIBRAÇÃO: se os resultados estiverem errados, execute:
#       python services/calibrate.py caminho/para/folha.jpg
#    O script gera debug_calibrate.jpg mostrando a grade detectada.
#    Ajuste os valores abaixo até a grade cobrir as bolhas corretamente.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Objetivas Q03–Q08 ────────────────────────────────────────────────────────
# Cada questão ocupa uma célula em grade 2 colunas × 3 linhas.
# Coluna esquerda: Q03, Q05, Q07 | Coluna direita: Q04, Q06, Q08

OBJ_ALTS = 5           # alternativas por questão (A, B, C, D, E)
OBJ_ALT_LABELS = ['A', 'B', 'C', 'D', 'E']

# Faixa Y de cada linha de objetivas [y_topo, y_base] normalizados
OBJ_ROWS_Y = [
    (0.37, 0.47),   # Linha 0: Q03 (esq) e Q04 (dir)
    (0.49, 0.59),   # Linha 1: Q05 (esq) e Q06 (dir)
    (0.61, 0.71),   # Linha 2: Q07 (esq) e Q08 (dir)
]

# Faixa X de cada coluna [x_esq, x_dir] normalizados
OBJ_COLS_X = [
    (0.03, 0.47),   # Coluna esquerda
    (0.53, 0.97),   # Coluna direita
]

# Mapeamento: (linha, coluna) → número da questão
OBJ_QUESTION_MAP = {
    (0, 0): 3,  (0, 1): 4,
    (1, 0): 5,  (1, 1): 6,
    (2, 0): 7,  (2, 1): 8,
}

# ── Somatórias Q09–Q10 ───────────────────────────────────────────────────────
# Cada somatória ocupa uma coluna, com duas fileiras de 10 bolhas (0–9).

SOMA_DIGITS = 10   # bolhas por fileira (0–9)

# Faixa Y para as fileiras Dez e Uni normalizados [y_topo, y_base]
SOMA_DEZ_Y = (0.76, 0.84)
SOMA_UNI_Y = (0.86, 0.94)

# Faixa X de cada questão somatória
SOMA_COLS_X = [
    (0.03, 0.47),   # Q09
    (0.53, 0.97),   # Q10
]

SOMA_QUESTION_MAP = {0: 9, 1: 10}

# ── Detecção de bolhas ───────────────────────────────────────────────────────
# Uma bolha é considerada PREENCHIDA se a intensidade média dos pixels na
# região for menor que este limiar (0–255, quanto menor, mais escuro).
FILL_INTENSITY_THRESHOLD = 160

# Contraste mínimo (diferença de intensidade) entre a bolha mais escura e a
# mais clara do grupo. Evita marcação quando a folha está em branco.
MIN_CONTRAST_OBJ = 25

# Limiar de preenchimento para somatórias (bolha preenchida se intensidade < limiar)
FILL_THRESHOLD_SOMA = 140


# ═══════════════════════════════════════════════════════════════════════════════
# DETECÇÃO DA FOLHA E PERSPECTIVA
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny com limiares automáticos baseados na mediana da imagem."""
    median = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos: topo-esq, topo-dir, baixo-dir, baixo-esq."""
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # menor soma → topo-esq
    rect[2] = pts[np.argmax(s)]   # maior soma → baixo-dir
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # menor diff → topo-dir
    rect[3] = pts[np.argmax(diff)]  # maior diff → baixo-esq
    return rect


def _detect_sheet_corners(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Tenta detectar o contorno retangular da folha de resposta.
    Retorna os 4 cantos ordenados, ou None se não encontrar.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Reduz resolução para processar mais rápido
    scale = min(1.0, 1200 / max(h, w))
    work = cv2.resize(gray, (int(w * scale), int(h * scale)))

    # Suavização e detecção de bordas
    blurred = cv2.GaussianBlur(work, (5, 5), 0)
    edges = _auto_canny(blurred)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    min_area = 0.15 * work.shape[0] * work.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            corners = _order_corners(approx) / scale
            return corners.astype(np.float32)

    return None


def _warp_sheet(img_bgr: np.ndarray, corners: Optional[np.ndarray]) -> np.ndarray:
    """
    Aplica transformação de perspectiva.
    Se corners for None, redimensiona diretamente (folha já alinhada).
    """
    if corners is None:
        return cv2.resize(img_bgr, (WARP_W, WARP_H))

    dst = np.array([
        [0, 0],
        [WARP_W - 1, 0],
        [WARP_W - 1, WARP_H - 1],
        [0, WARP_H - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img_bgr, M, (WARP_W, WARP_H))


# ═══════════════════════════════════════════════════════════════════════════════
# PRÉ-PROCESSAMENTO
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess(warped_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna (gray_normalized, thresh) para análise de bolhas.
    Usa CLAHE para normalizar iluminação de fotos com sombras.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE: equalização de histograma adaptativa por região
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Threshold adaptativo — mais robusto que Otsu para fotos com iluminação variável
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=10
    )

    return gray, thresh


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISE DE INTENSIDADE POR REGIÃO
# ═══════════════════════════════════════════════════════════════════════════════

def _roi_pixels(x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
    """Converte coordenadas normalizadas para pixels no canvas warped."""
    return (
        int(x0 * WARP_W), int(y0 * WARP_H),
        int(x1 * WARP_W), int(y1 * WARP_H)
    )


def _sample_bubble_intensity(gray: np.ndarray, cx_px: int, cy_px: int, r_px: int) -> float:
    """
    Amostra a intensidade média dentro de um círculo em (cx_px, cy_px) com raio r_px.
    Retorna 0–255 (menor = mais escuro = mais preenchido).
    """
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx_px, cy_px), r_px, 255, -1)
    pixels = gray[mask > 0]
    return float(np.mean(pixels)) if len(pixels) > 0 else 255.0


def _read_bubble_row(
    gray: np.ndarray,
    x0_n: float, y0_n: float, x1_n: float, y1_n: float,
    n_bubbles: int,
) -> list[float]:
    """
    Lê uma fileira horizontal de n_bubbles bolhas dentro da região normalizada.
    Retorna lista de intensidades médias (0–255) para cada posição.
    """
    px0, py0, px1, py1 = _roi_pixels(x0_n, y0_n, x1_n, y1_n)
    row_h = py1 - py0
    row_w = px1 - px0

    # Raio estimado da bolha: metade da altura da região, com margem
    r = max(4, int(row_h * 0.35))

    # Centros das bolhas: distribuídos horizontalmente com margem
    margin = int(row_w * 0.04)
    step = (row_w - 2 * margin) / max(n_bubbles - 1, 1)
    cy = py0 + row_h // 2

    intensities = []
    for i in range(n_bubbles):
        cx = px0 + margin + int(i * step)
        intensity = _sample_bubble_intensity(gray, cx, cy, r)
        intensities.append(intensity)

    return intensities


# ═══════════════════════════════════════════════════════════════════════════════
# LEITURA DAS QUESTÕES
# ═══════════════════════════════════════════════════════════════════════════════

def _read_objective_questions(gray: np.ndarray) -> dict[int, Optional[str]]:
    """
    Lê as 6 questões objetivas (Q03–Q08).
    Retorna {3: 'A', 4: None, ...}  (None = em branco ou ilegível)
    """
    results: dict[int, Optional[str]] = {}

    for row_idx, (y0_n, y1_n) in enumerate(OBJ_ROWS_Y):
        for col_idx, (x0_n, x1_n) in enumerate(OBJ_COLS_X):
            q_num = OBJ_QUESTION_MAP[(row_idx, col_idx)]

            intensities = _read_bubble_row(gray, x0_n, y0_n, x1_n, y1_n, OBJ_ALTS)

            # A bolha selecionada é a mais escura
            min_intensity = min(intensities)
            max_intensity = max(intensities)
            contrast = max_intensity - min_intensity

            if contrast < MIN_CONTRAST_OBJ or min_intensity > FILL_INTENSITY_THRESHOLD:
                # Não há bolha claramente marcada
                results[q_num] = None
            else:
                selected_idx = int(np.argmin(intensities))
                results[q_num] = OBJ_ALT_LABELS[selected_idx]

    return results


def _read_summation_questions(gray: np.ndarray) -> dict[int, Optional[int]]:
    """
    Lê as 2 questões somatórias (Q09–Q10).
    Cada questão tem fileira Dez (0–9) e fileira Uni (0–9).
    Retorna {9: 20, 10: 10, ...}  (None = ilegível)
    """
    results: dict[int, Optional[int]] = {}

    for col_idx, (x0_n, x1_n) in enumerate(SOMA_COLS_X):
        q_num = SOMA_QUESTION_MAP[col_idx]

        # Fileira das dezenas
        dez_intensities = _read_bubble_row(
            gray, x0_n, SOMA_DEZ_Y[0], x1_n, SOMA_DEZ_Y[1], SOMA_DIGITS
        )
        # Fileira das unidades
        uni_intensities = _read_bubble_row(
            gray, x0_n, SOMA_UNI_Y[0], x1_n, SOMA_UNI_Y[1], SOMA_DIGITS
        )

        def pick_digit(intensities: list[float]) -> Optional[int]:
            min_val = min(intensities)
            max_val = max(intensities)
            if max_val - min_val < MIN_CONTRAST_OBJ:
                return None
            if min_val > FILL_THRESHOLD_SOMA:
                return None
            return int(np.argmin(intensities))

        dez = pick_digit(dez_intensities)
        uni = pick_digit(uni_intensities)

        if dez is None or uni is None:
            results[q_num] = None
        else:
            results[q_num] = dez * 10 + uni

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CORREÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

def _correct(
    omr_result: dict,
    answer_key: dict,
) -> dict:
    """
    Aplica o gabarito e retorna a correção.
    answer_key esperado: {
        "obj": {"3": "A", "4": "B", ...},
        "soma": {"9": 20, "10": 10}
    }
    """
    resultado = []
    acertos = 0
    erros = 0

    obj_gab = answer_key.get("obj", {})
    soma_gab = answer_key.get("soma", {})

    obj_aluno = omr_result.get("objetivas", {})
    soma_aluno = omr_result.get("somatorias", {})

    for q_num in [3, 4, 5, 6, 7, 8]:
        q_str = str(q_num)
        gab = obj_gab.get(q_str)
        alu = obj_aluno.get(q_num)
        correto = gab is not None and alu == gab
        if correto:
            acertos += 1
        else:
            erros += 1
        resultado.append({
            "questao": f"Q{q_num:02d}",
            "tipo": "objetiva",
            "gabarito": gab,
            "aluno": alu,
            "correto": correto,
        })

    for q_num in [9, 10]:
        q_str = str(q_num)
        gab = soma_gab.get(q_str)
        alu = soma_aluno.get(q_num)
        correto = gab is not None and alu is not None and int(alu) == int(gab)
        if correto:
            acertos += 1
        else:
            erros += 1
        resultado.append({
            "questao": f"Q{q_num:02d}",
            "tipo": "somatoria",
            "gabarito": gab,
            "aluno": alu,
            "correto": correto,
        })

    total_questoes = 8
    nota_total = round(acertos / total_questoes * 10, 1)

    return {
        "nota_total": nota_total,
        "acertos": acertos,
        "erros": erros,
        "resultado": resultado,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def process_answer_sheet(
    image_path: str,
    answer_key: Optional[dict] = None,
    debug: bool = False,
    debug_path: str = "debug_omr.jpg",
) -> dict:
    """
    Processa uma folha de resposta e retorna as respostas detectadas.

    Args:
        image_path:  Caminho para a imagem (JPG, PNG, WEBP).
        answer_key:  Gabarito opcional. Se fornecido, retorna também a correção.
                     Formato: {"obj": {"3":"A",...}, "soma": {"9":20,...}}
        debug:       Se True, salva imagem anotada em debug_path.
        debug_path:  Caminho para salvar a imagem de debug.

    Returns:
        Sem gabarito:
            {"objetivas": {3:"A", 4:"B", ...}, "somatorias": {9:20, 10:10}}
        Com gabarito:
            {"objetivas": ..., "somatorias": ...,
             "nota_total": 7.5, "acertos": 6, "erros": 2, "resultado": [...]}
        Em caso de erro:
            {"erro": "mensagem"}
    """
    # ── 1. Carregar imagem ────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        return {"erro": f"Não foi possível abrir a imagem: {image_path}"}

    # Reduz imagens muito grandes antes de processar (economiza memória)
    h, w = img.shape[:2]
    if max(h, w) > 2400:
        scale = 2400 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # ── 2. Detectar bordas da folha ───────────────────────────────────────────
    corners = _detect_sheet_corners(img)

    # ── 3. Warp perspectivo ───────────────────────────────────────────────────
    warped = _warp_sheet(img, corners)

    # ── 4. Pré-processamento ──────────────────────────────────────────────────
    gray, thresh = _preprocess(warped)

    # ── 5. Leitura das questões ───────────────────────────────────────────────
    objetivas = _read_objective_questions(gray)
    somatorias = _read_summation_questions(gray)

    result = {
        "objetivas": objetivas,
        "somatorias": somatorias,
        "sheet_detected": corners is not None,
    }

    # ── 6. Correção (se gabarito fornecido) ───────────────────────────────────
    if answer_key:
        correction = _correct(result, answer_key)
        result.update(correction)

    # ── 7. Debug ──────────────────────────────────────────────────────────────
    if debug:
        _draw_debug(warped, gray, debug_path)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DEBUG — visualização da grade de bolhas
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_debug(warped: np.ndarray, gray: np.ndarray, path: str) -> None:
    """
    Salva imagem com a grade de bolhas anotada para calibração visual.
    Círculos verdes = bolhas detectadas como preenchidas.
    Círculos vermelhos = bolhas vazias.
    """
    annotated = warped.copy()

    def draw_row(x0_n, y0_n, x1_n, y1_n, n, label=""):
        px0, py0, px1, py1 = _roi_pixels(x0_n, y0_n, x1_n, y1_n)
        row_h = py1 - py0
        row_w = px1 - px0
        r = max(4, int(row_h * 0.35))
        margin = int(row_w * 0.04)
        step = (row_w - 2 * margin) / max(n - 1, 1)
        cy = py0 + row_h // 2
        cv2.rectangle(annotated, (px0, py0), (px1, py1), (200, 200, 0), 1)
        for i in range(n):
            cx = px0 + margin + int(i * step)
            intensity = _sample_bubble_intensity(gray, cx, cy, r)
            filled = intensity < FILL_INTENSITY_THRESHOLD
            color = (0, 200, 0) if filled else (0, 0, 200)
            cv2.circle(annotated, (cx, cy), r, color, 2)
            cv2.putText(annotated, str(i) if n > 5 else ['A','B','C','D','E'][i],
                        (cx - 6, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, color, 1, cv2.LINE_AA)
        if label:
            cv2.putText(annotated, label, (px0, py0 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 80, 0), 1)

    # Objetivas
    for row_idx, (y0, y1) in enumerate(OBJ_ROWS_Y):
        for col_idx, (x0, x1) in enumerate(OBJ_COLS_X):
            q = OBJ_QUESTION_MAP[(row_idx, col_idx)]
            draw_row(x0, y0, x1, y1, OBJ_ALTS, f"Q{q:02d}")

    # Somatórias
    for col_idx, (x0, x1) in enumerate(SOMA_COLS_X):
        q = SOMA_QUESTION_MAP[col_idx]
        draw_row(x0, SOMA_DEZ_Y[0], x1, SOMA_DEZ_Y[1], SOMA_DIGITS, f"Q{q:02d} Dez")
        draw_row(x0, SOMA_UNI_Y[0], x1, SOMA_UNI_Y[1], SOMA_DIGITS, f"Q{q:02d} Uni")

    cv2.imwrite(path, annotated)
    print(f"[DEBUG] Grade anotada salva em: {path}")
