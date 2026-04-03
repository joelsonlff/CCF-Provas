"""
calibrate.py
============
Ferramenta de calibração para o OMR processor.

Uso:
    python services/calibrate.py caminho/para/folha.jpg

O script:
  1. Detecta a folha e aplica warp perspectivo
  2. Salva o canvas warped em debug_warped.jpg
  3. Desenha a grade de bolhas atual em debug_calibrate.jpg
  4. Imprime as intensidades de cada posição para diagnóstico

Se as bolhas não estiverem alinhadas com os círculos na imagem de debug,
ajuste as constantes OBJ_ROWS_Y, OBJ_COLS_X, SOMA_DEZ_Y, SOMA_UNI_Y
em services/omr_processor.py até que a grade se encaixe.
"""

import sys
import os
import cv2
import numpy as np

# Adiciona o diretório pai ao path para importar o módulo corretamente
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.omr_processor import (
    _detect_sheet_corners,
    _warp_sheet,
    _preprocess,
    _read_objective_questions,
    _read_summation_questions,
    _draw_debug,
    _read_bubble_row,
    OBJ_ROWS_Y, OBJ_COLS_X, OBJ_ALTS, OBJ_QUESTION_MAP,
    SOMA_BUBBLES_X,
    SOMA_Q09_DEZ_Y, SOMA_Q09_UNI_Y,
    SOMA_Q10_DEZ_Y, SOMA_Q10_UNI_Y,
    SOMA_DIGITS, SOMA_QUESTION_MAP,
    WARP_W, WARP_H,
)


def run_calibration(image_path: str) -> None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERRO: Não foi possível abrir '{image_path}'")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"\n[Calibração] Imagem: {image_path}  ({w}×{h} px)")

    # ── Detecção da folha ─────────────────────────────────────────────────────
    corners = _detect_sheet_corners(img)
    if corners is not None:
        print(f"[OK] Folha detectada — cantos: {corners.tolist()}")
    else:
        print("[AVISO] Folha NÃO detectada. Usando imagem inteira (sem warp).")
        print("        Dica: melhore o contraste entre a folha e o fundo.")

    # ── Warp ──────────────────────────────────────────────────────────────────
    warped = _warp_sheet(img, corners)
    cv2.imwrite("debug_warped.jpg", warped)
    print(f"[OK] Canvas warped ({WARP_W}×{WARP_H}) salvo em: debug_warped.jpg")

    # ── Pré-processamento ─────────────────────────────────────────────────────
    gray, thresh = _preprocess(warped)

    # ── Imprimir intensidades das objetivas ───────────────────────────────────
    print("\n── OBJETIVAS ────────────────────────────────────────")
    print("  (menor intensidade = mais escuro = bolha marcada)")
    for row_idx, (y0, y1) in enumerate(OBJ_ROWS_Y):
        for col_idx, (x0, x1) in enumerate(OBJ_COLS_X):
            q = OBJ_QUESTION_MAP[(row_idx, col_idx)]
            intensities = _read_bubble_row(gray, x0, y0, x1, y1, OBJ_ALTS)
            labels = ['A', 'B', 'C', 'D', 'E']
            formatted = "  ".join(
                f"{labels[i]}:{v:5.1f}{'*' if v == min(intensities) else ' '}"
                for i, v in enumerate(intensities)
            )
            min_v = min(intensities)
            max_v = max(intensities)
            contrast = max_v - min_v
            print(f"  Q{q:02d}: {formatted}  |contraste={contrast:.1f}")

    # ── Imprimir intensidades das somatórias ──────────────────────────────────
    print("\n── SOMATÓRIAS ───────────────────────────────────────")
    bx0, bx1 = SOMA_BUBBLES_X
    soma_layout_cal = [
        (9,  SOMA_Q09_DEZ_Y, SOMA_Q09_UNI_Y),
        (10, SOMA_Q10_DEZ_Y, SOMA_Q10_UNI_Y),
    ]
    for q, dez_y, uni_y in soma_layout_cal:
        x0, x1 = bx0, bx1
        dez = _read_bubble_row(gray, x0, dez_y[0], x1, dez_y[1], SOMA_DIGITS)
        uni = _read_bubble_row(gray, x0, uni_y[0], x1, uni_y[1], SOMA_DIGITS)

        fmt_dez = "  ".join(
            f"{i}:{v:5.1f}{'*' if v == min(dez) else ' '}"
            for i, v in enumerate(dez)
        )
        fmt_uni = "  ".join(
            f"{i}:{v:5.1f}{'*' if v == min(uni) else ' '}"
            for i, v in enumerate(uni)
        )
        print(f"  Q{q:02d} Dez: {fmt_dez}")
        print(f"  Q{q:02d} Uni: {fmt_uni}")

    # ── Resultado detectado ───────────────────────────────────────────────────
    obj = _read_objective_questions(gray)
    soma = _read_summation_questions(gray)

    print("\n── RESULTADO DETECTADO ──────────────────────────────")
    for q_num in sorted(obj):
        print(f"  Q{q_num:02d}: {obj[q_num]}")
    for q_num in sorted(soma):
        val = soma[q_num]
        if val is not None:
            print(f"  Q{q_num:02d}: {val}  (Dez={val//10}, Uni={val%10})")
        else:
            print(f"  Q{q_num:02d}: (não detectado)")

    # ── Debug visual ──────────────────────────────────────────────────────────
    _draw_debug(warped, gray, "debug_calibrate.jpg")
    print("\n[OK] Grade anotada salva em: debug_calibrate.jpg")
    print("\nVerifique se os círculos na imagem estão sobre as bolhas da folha.")
    print("Se não estiverem, ajuste os valores em services/omr_processor.py:\n")
    print("  OBJ_ROWS_Y  — faixas Y das linhas de objetivas")
    print("  OBJ_COLS_X  — faixas X das colunas (esq/dir)")
    print("  SOMA_DEZ_Y  — faixa Y da fileira Dez")
    print("  SOMA_UNI_Y  — faixa Y da fileira Uni")
    print("  SOMA_COLS_X — faixas X das somatórias (Q09/Q10)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python services/calibrate.py caminho/para/folha.jpg")
        sys.exit(1)
    run_calibration(sys.argv[1])
