"""
omr_processor.py
================
OMR (Optical Mark Recognition) local para folhas de resposta CCF.
Detecta bolhas preenchidas usando OpenCV — sem chamadas a APIs externas.

Layout real da folha CCF (de cima para baixo):
  [Cabeçalho]       Logo + título + QR REF-1 + campos (aluno, turma, etc.)
  [Q01–Q02]         Questões discursivas em caixas próprias — ignoradas
  [QUESTÕES OBJETIVAS]  barra de seção
  [Q03–Q08]         Grade 2 colunas × 3 linhas, 5 bolhas (A–E) por questão
  [QUESTÕES SOMATÓRIAS] barra de seção
  [Q09]             Caixa full-width:  Dez: 0–9  /  Uni: 0–9
  [Q10]             Caixa full-width:  Dez: 0–9  /  Uni: 0–9  + QR REF-3
  [Rodapé]          QR REF-2 + texto + caixa NOTA

Resultado somatória = Dez × 10 + Uni  (inteiro 0–99)

Uso:
    from services.omr_processor import process_answer_sheet
    result = process_answer_sheet("foto.jpg")
    # {"objetivas": {3:"A",...}, "somatorias": {9:20, 10:10}}

Calibração:
    python services/calibrate.py imagens_provas/foto.jpg
    → debug_warped.jpg e debug_calibrate.jpg para verificar o alinhamento.
    Ajuste as constantes de LAYOUT abaixo se necessário.
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
# Derivadas da leitura visual do PDF da folha CCF real.
# ⚠  Se os resultados estiverem errados, rode:
#       python services/calibrate.py imagens_provas/foto.jpg
#    e ajuste os valores abaixo até a grade cobrir as bolhas.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Objetivas Q03–Q08 ────────────────────────────────────────────────────────
# Grade 2 colunas × 3 linhas. Coluna esq: Q03,Q05,Q07 | dir: Q04,Q06,Q08.
# As faixas X aqui JÁ excluem o rótulo "Q0X" à esquerda de cada célula.
# Calibradas visualmente a partir do PDF/JPG real da folha CCF.

OBJ_ALTS = 5
OBJ_ALT_LABELS = ['A', 'B', 'C', 'D', 'E']

# Faixa Y de cada linha [y_topo, y_base]
# Calibradas com HoughCircles na folha v5 (centros medidos em px/1100):
#   Linha 0: cy≈595  |  Linha 1: cy≈647  |  Linha 2: cy≈700
OBJ_ROWS_Y = [
    (0.520, 0.562),  # Linha 0: Q03/Q04  cy≈595 (0.541)
    (0.567, 0.609),  # Linha 1: Q05/Q06  cy≈647 (0.588)
    (0.615, 0.657),  # Linha 2: Q07/Q08  cy≈700 (0.636)
]

# Faixa X das BOLHAS (HoughCircles folha v5):
#   Coluna esq: cx_min=0.119, cx_max=0.346, r≈0.020 → margem 0.099..0.366
#   Coluna dir: cx_min=0.604, cx_max=0.830, r≈0.020 → margem 0.584..0.850
OBJ_COLS_X = [
    (0.112, 0.366),  # Coluna esquerda: bolhas A–E  (cx_A=0.119, margem -r)
    (0.598, 0.850),  # Coluna direita:  bolhas A–E  (cx_A=0.604, margem -r)
]

# Mapeamento: (linha, coluna) → número da questão
OBJ_QUESTION_MAP = {
    (0, 0): 3,  (0, 1): 4,
    (1, 0): 5,  (1, 1): 6,
    (2, 0): 7,  (2, 1): 8,
}

# ── Somatórias Q09–Q10 ───────────────────────────────────────────────────────
# Q09 e Q10 são BLOCOS FULL-WIDTH empilhados verticalmente (NÃO em colunas).
# Cada bloco tem 2 fileiras: Dez (0–9) e Uni (0–9).

SOMA_DIGITS = 10  # bolhas por fileira (0–9)

# Faixa X das bolhas (medida com HoughCircles na folha v4):
#   10 bolhas: cx=97..424, r≈16 → (97-20)/800=0.096 .. (424+20)/800=0.555
#   Rótulos "Dez:"/"Uni:" ocupam aprox x=0..0.096, começa logo depois.
# Faixa X das bolhas somatórias (HoughCircles folha v5):
#   cx_min=0.119, cx_max=0.606, r≈0.021 → margem 0.098..0.627
SOMA_BUBBLES_X = (0.098, 0.627)

# Q09 — Dez: cy≈812 (0.738) | Uni: cy≈843 (0.766)
# Bandas estreitas e sem sobreposição (gap = 2px entre elas)
SOMA_Q09_DEZ_Y = (0.719, 0.752)   # centro 0.738, ±0.019
SOMA_Q09_UNI_Y = (0.754, 0.787)   # centro 0.766, ±0.019 (começa após Dez)

# Q10 — Dez: cy≈945 (0.859) | Uni: cy≈976 (0.887)
SOMA_Q10_DEZ_Y = (0.840, 0.875)   # centro 0.859, ±0.019
SOMA_Q10_UNI_Y = (0.877, 0.910)   # centro 0.887, ±0.019 (começa após Dez)

SOMA_QUESTION_MAP = {0: 9, 1: 10}

# ── Limiares de detecção ─────────────────────────────────────────────────────
# Intensidade máxima para considerar uma bolha PREENCHIDA (0–255).
# Bolhas vazias têm intensidade média ~210–240 (branco).
# Bolhas preenchidas a caneta azul/preta: ~70–110 (escuro).
# Valor 125 evita falsos positivos de bordas de bolhas (~147) e
# garante detecção de bolhas com tinta (~70–110).
FILL_INTENSITY_THRESHOLD = 135

# Contraste mínimo entre a bolha mais escura e a mais clara do grupo.
# Num gabarito em branco: contraste real das bolhas ~16–20 (ruído de borda).
# Numa bolha preenchida: contraste ~80–150.
# Valor 40 evita falsos positivos em gabaritos em branco.
MIN_CONTRAST_OBJ = 40

# Limiar para somatórias (igual ao das objetivas após a correção do QR code)
FILL_THRESHOLD_SOMA = 135


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
    Detecta os 4 cantos da folha de resposta.

    Estratégia 1 (preferida): localiza os 4 marcadores fiduciais pretos
    (quadrados sólidos ~8×8mm nos cantos da folha v4+).

    Estratégia 2 (fallback): detecta o maior contorno retangular da imagem
    (funciona quando há contraste entre a folha e o fundo).

    Retorna array float32 com 4 pontos ordenados [topo-esq, topo-dir,
    baixo-dir, baixo-esq], ou None se nenhuma estratégia funcionar.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Estratégia 1: marcadores fiduciais pretos nos cantos ─────────────────
    # Tenta limiares progressivamente mais flexíveis (60 → 80 → 100)
    # para lidar com fotos mais claras ou marcadores pouco contrastados.
    min_marker_px = int(min(h, w) * 0.012)
    max_marker_px = int(min(h, w) * 0.07)

    def _find_marker_candidates(threshold: int):
        _, dark_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if not (min_marker_px < cw < max_marker_px and min_marker_px < ch < max_marker_px):
                continue
            if not (0.5 < cw / ch < 2.0):
                continue
            if cv2.contourArea(cnt) < min_marker_px * min_marker_px * 0.3:
                continue
            found.append((x + cw // 2, y + ch // 2, cv2.contourArea(cnt)))
        return found

    def _pick_quadrant_markers(candidates):
        cx_mid, cy_mid = w / 2, h / 2
        quads = {
            'tl': [c for c in candidates if c[0] < cx_mid and c[1] < cy_mid],
            'tr': [c for c in candidates if c[0] >= cx_mid and c[1] < cy_mid],
            'br': [c for c in candidates if c[0] >= cx_mid and c[1] >= cy_mid],
            'bl': [c for c in candidates if c[0] < cx_mid and c[1] >= cy_mid],
        }
        pick = lambda q: max(q, key=lambda c: c[2]) if q else None
        return {k: pick(v) for k, v in quads.items()}

    best_candidates = []
    for thresh in (60, 80, 100):
        cands = _find_marker_candidates(thresh)
        if len(cands) > len(best_candidates):
            best_candidates = cands

    quads = _pick_quadrant_markers(best_candidates)
    tl, tr, br, bl = quads['tl'], quads['tr'], quads['br'], quads['bl']

    if tl and tr and br and bl:
        # Estratégia 1a: todos os 4 marcadores encontrados
        corners = np.float32([
            [tl[0], tl[1]], [tr[0], tr[1]],
            [br[0], br[1]], [bl[0], bl[1]],
        ])
        return corners

    # Estratégia 1b: apenas 3 marcadores → estima o 4º pelo paralelogramo
    # Propriedade: TL + BR = TR + BL  (diagonais se bissectam no centro)
    present = {k: v for k, v in quads.items() if v is not None}
    if len(present) == 3:
        missing = [k for k in ('tl', 'tr', 'br', 'bl') if quads[k] is None][0]
        p = {k: np.float32([v[0], v[1]]) for k, v in present.items()}
        if missing == 'tl':   estimated = p['tr'] + p['bl'] - p['br']
        elif missing == 'tr': estimated = p['tl'] + p['br'] - p['bl']
        elif missing == 'br': estimated = p['tr'] + p['bl'] - p['tl']
        else:                 estimated = p['tl'] + p['br'] - p['tr']
        quads[missing] = (float(estimated[0]), float(estimated[1]), 0)
        tl, tr, br, bl = quads['tl'], quads['tr'], quads['br'], quads['bl']
        corners = np.float32([
            [tl[0], tl[1]], [tr[0], tr[1]],
            [br[0], br[1]], [bl[0], bl[1]],
        ])
        return corners

    # ── Estratégia 2: maior contorno retangular (folha vs fundo contrastante) ─
    scale = min(1.0, 1200 / max(h, w))
    work = cv2.resize(gray, (int(w * scale), int(h * scale)))
    blurred = cv2.GaussianBlur(work, (5, 5), 0)
    edges = _auto_canny(blurred)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:10]
    min_area = 0.15 * work.shape[0] * work.shape[1]

    for cnt in contours2:
        if cv2.contourArea(cnt) < min_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return (_order_corners(approx) / scale).astype(np.float32)

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
    Q09 e Q10 são blocos full-width empilhados verticalmente.
    Cada bloco tem fileira Dez (0–9) e Uni (0–9).
    Retorna {9: 20, 10: 10}  (None = ilegível)
    """
    # Layout: (q_num, dez_y_range, uni_y_range)
    soma_layout = [
        (9,  SOMA_Q09_DEZ_Y, SOMA_Q09_UNI_Y),
        (10, SOMA_Q10_DEZ_Y, SOMA_Q10_UNI_Y),
    ]

    x0_n, x1_n = SOMA_BUBBLES_X

    def pick_digit(intensities: list[float]) -> Optional[int]:
        min_val = min(intensities)
        max_val = max(intensities)
        if max_val - min_val < MIN_CONTRAST_OBJ:
            return None
        if min_val > FILL_THRESHOLD_SOMA:
            return None
        return int(np.argmin(intensities))

    results: dict[int, Optional[int]] = {}
    for q_num, dez_y, uni_y in soma_layout:
        dez_intensities = _read_bubble_row(
            gray, x0_n, dez_y[0], x1_n, dez_y[1], SOMA_DIGITS
        )
        uni_intensities = _read_bubble_row(
            gray, x0_n, uni_y[0], x1_n, uni_y[1], SOMA_DIGITS
        )
        dez = pick_digit(dez_intensities)
        uni = pick_digit(uni_intensities)
        results[q_num] = (dez * 10 + uni) if (dez is not None and uni is not None) else None

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

    # Somatórias (full-width, empilhadas)
    x0, x1 = SOMA_BUBBLES_X
    draw_row(x0, SOMA_Q09_DEZ_Y[0], x1, SOMA_Q09_DEZ_Y[1], SOMA_DIGITS, "Q09 Dez")
    draw_row(x0, SOMA_Q09_UNI_Y[0], x1, SOMA_Q09_UNI_Y[1], SOMA_DIGITS, "Q09 Uni")
    draw_row(x0, SOMA_Q10_DEZ_Y[0], x1, SOMA_Q10_DEZ_Y[1], SOMA_DIGITS, "Q10 Dez")
    draw_row(x0, SOMA_Q10_UNI_Y[0], x1, SOMA_Q10_UNI_Y[1], SOMA_DIGITS, "Q10 Uni")

    cv2.imwrite(path, annotated)
    print(f"[DEBUG] Grade anotada salva em: {path}")
