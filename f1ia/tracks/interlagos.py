"""Traçado ESTILIZADO inspirado no Autódromo José Carlos Pace (Interlagos), ~4,3 km, anti-horário.

Não são coordenadas de GPS: os pontos de controle foram desenhados à mão para reproduzir a
sequência de setores (Senna S, Curva do Sol, Reta Oposta, Descida do Lago, Ferradura,
Laranjinha, Pinheirinho, Bico de Pato, Mergulho, Junção, Subida dos Boxes) e a escala.
Para usar dados reais, exporte a linha central (x,y em metros) para CSV e use
`--tracks csv:caminho.csv`.
"""
from __future__ import annotations

import numpy as np

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

SECTORS = [
    ("Reta dos Boxes", 0.00), ("Senna S", 0.10), ("Curva do Sol", 0.18), ("Reta Oposta", 0.30),
    ("Descida do Lago", 0.47), ("Ferradura", 0.53), ("Laranjinha", 0.62), ("Pinheirinho", 0.68),
    ("Bico de Pato", 0.78), ("Mergulho", 0.87), ("Junção", 0.93), ("Subida dos Boxes", 0.97),
]


LENGTH_M = 4309.0        # extensão oficial
RACE_LAPS = 71           # 71 x 4,309 km = 305,939 km
ELEVATION_DROP = 56.0    # desnível aproximado entre o ponto mais alto e o mais baixo (m)
# (fração da volta, altitude em m): alto na reta dos boxes, mais baixo perto do Pinheirinho
ELEVATION = [(0.00, 800), (0.08, 797), (0.12, 785), (0.18, 776), (0.30, 770), (0.45, 774), (0.50, 765),
             (0.53, 750), (0.62, 746), (0.68, 744), (0.78, 752), (0.87, 760), (0.93, 773), (0.97, 788)]


def _nearest_s(track, xy):
    return float(track.s[np.linalg.norm(track.center - np.asarray(xy), axis=1).argmin()])


def build_interlagos() -> Track:
    scale = 1.263
    for _ in range(6):   # ajusta a escala até a extensão ficar em 4.309 m
        pts = [(x * scale, y * scale) for x, y in CONTROL_POINTS]
        track = Track.from_control_points("Interlagos (estilizada)", pts, TRACK_WIDTH)
        if abs(track.length - LENGTH_M) < 0.05:
            break
        scale *= LENGTH_M / track.length
    track.set_elevation([f for f, _ in ELEVATION], [e for _, e in ELEVATION], drop=ELEVATION_DROP)
    track.elevation += 765.0 - track.elevation.mean() + 0  # altitude média ~765 m
    track.elevation += 0.0
    near = lambda i: _nearest_s(track, (CONTROL_POINTS[i][0] * scale, CONTROL_POINTS[i][1] * scale))
    # DRS 1: reta dos boxes (saída da Junção até o Senna S) | DRS 2: reta oposta
    track.set_drs([(near(len(CONTROL_POINTS) - 2), near(3)), (near(12), near(15))])
    track.meta = {"circuit": "Autódromo José Carlos Pace (estilizado)", "laps": RACE_LAPS,
                  "race_km": round(RACE_LAPS * track.length / 1000, 2), "elevation_drop_m": ELEVATION_DROP,
                  "altitude_avg_m": round(float(track.elevation.mean())), "drs_zones": 2}
    return track
