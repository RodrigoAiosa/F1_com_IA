"""Traçado ESTILIZADO inspirado no Autódromo José Carlos Pace (Interlagos), ~4,3 km, anti-horário.

Não são coordenadas de GPS: os pontos de controle foram desenhados à mão para reproduzir a
sequência de setores (Senna S, Curva do Sol, Reta Oposta, Descida do Lago, Ferradura,
Laranjinha, Pinheirinho, Bico de Pato, Mergulho, Junção, Subida dos Boxes) e a escala.
Para usar dados reais, exporte a linha central (x,y em metros) para CSV e use
`--tracks csv:caminho.csv`.
"""
from __future__ import annotations

from ..track import Track

# (x, y) em metros, sentido anti-horário. O primeiro ponto fica na reta dos boxes.
CONTROL_POINTS = [
    # Reta dos boxes (para oeste)
    (700, 330), (520, 330), (380, 322),
    # Senna S (esquerda-direita)
    (300, 290), (262, 235), (270, 175),
    # Curva do Sol (esquerda longa)
    (240, 110), (170, 70), (110, 20), (110, -60), (170, -120), (260, -150),
    # Reta Oposta (para leste)
    (420, -168), (640, -170), (860, -160), (1040, -140),
    # Descida do Lago (esquerda) -> Ferradura (direita longa)
    (1150, -105), (1190, -40), (1160, 20), (1085, 45),
    # Laranjinha (esquerda)
    (1000, 25), (930, -20),
    # Pinheirinho (grampo à esquerda)
    (850, -40), (800, 5), (835, 55), (900, 60),
    # Bico de Pato (esquerda-direita)
    (975, 100), (1030, 165), (1100, 205),
    # Mergulho (direita) -> Junção (esquerda)
    (1130, 270), (1090, 335), (1010, 350),
    # Subida dos Boxes
    (900, 345), (800, 335),
]

TRACK_WIDTH = 14.0
SCALE = 1.263  # ajusta o comprimento para ~4,3 km

SECTORS = [
    ("Reta dos Boxes", 0.00), ("Senna S", 0.10), ("Curva do Sol", 0.18), ("Reta Oposta", 0.30),
    ("Descida do Lago", 0.47), ("Ferradura", 0.53), ("Laranjinha", 0.62), ("Pinheirinho", 0.68),
    ("Bico de Pato", 0.78), ("Mergulho", 0.87), ("Junção", 0.93), ("Subida dos Boxes", 0.97),
]


def build_interlagos() -> Track:
    pts = [(x * SCALE, y * SCALE) for x, y in CONTROL_POINTS]
    return Track.from_control_points("Interlagos (estilizada)", pts, TRACK_WIDTH)
