"""Interlagos (Autódromo José Carlos Pace): traçado reproduzido do mapa oficial do GP de São Paulo.

A linha central foi traçada sobre a imagem do mapa (pixels, y para baixo) e depois convertida para
metros: a escala é ajustada para a extensão oficial de 4.309 m. É uma reprodução de planta a partir de
uma imagem, não dados topográficos/GPS. Para dados reais, use `--tracks csv:arquivo.csv` (x,y em m).
Sentido anti-horário; a largada/chegada fica na reta superior (bandeira quadriculada do mapa).
"""
from __future__ import annotations

import numpy as np

from ..track import Track

# Pontos da linha central em pixels do mapa (imagem ampliada 4x: 1288x724), na ordem da volta a partir
# da linha de chegada. Comentários = curvas numeradas do mapa.
MAP_POINTS_PX = [
    (462, 185),                                   # linha de largada, reta dos boxes (DRS 1)
    (370, 210), (300, 228), (255, 240),
    (225, 262), (232, 292), (262, 312),           # T1 / T2 (Senna S)
    (250, 340), (238, 390), (238, 445),
    (255, 495), (300, 528),                       # T3 (Curva do Sol)
    (380, 556), (480, 584), (580, 615), (690, 645), (790, 665),   # reta oposta (DRS 2)
    (835, 668), (865, 640), (870, 585),           # T4 (Descida do Lago)
    (850, 525), (790, 460), (720, 400), (650, 340),               # T5 -> subida para o laço
    (597, 295), (580, 250), (600, 205),           # T6
    (645, 182), (700, 175), (745, 182),           # T7
    (765, 210), (745, 240), (722, 268),           # T8
    (735, 292), (770, 298), (810, 275),           # T9
    (860, 235), (905, 205), (945, 205), (952, 230),  # T10
    (925, 262), (880, 300), (850, 335), (850, 370),  # T11
    (880, 420), (935, 465), (990, 492), (1030, 495),  # entrada da T12
    (1062, 470), (1077, 420), (1075, 375),        # T13
    (1062, 330), (1055, 285), (1038, 240),
    (1008, 200), (960, 172), (900, 150),          # T14
    (850, 143), (790, 138),                       # T15 -> reta dos boxes (início do DRS 1)
    (700, 140), (600, 160), (530, 172),
]
DRS_PX = [((790, 138), (275, 235)),   # DRS 1: reta dos boxes (cruza a linha de largada)
          ((320, 538), (795, 664))]   # DRS 2: reta oposta

TRACK_WIDTH = 14.0
LENGTH_M = 4309.0        # extensão oficial
RACE_LAPS = 71           # 71 x 4,309 km = 305,939 km
ELEVATION_DROP = 56.0    # desnível aproximado entre o ponto mais alto e o mais baixo (m)
ALTITUDE_AVG = 765.0
# (fração da volta, altitude em m): alto na reta dos boxes, mais baixo no laço interno/Descida do Lago
ELEVATION = [(0.00, 800), (0.08, 797), (0.12, 785), (0.18, 776), (0.30, 770), (0.45, 774), (0.50, 765),
             (0.53, 750), (0.62, 746), (0.68, 744), (0.78, 752), (0.87, 760), (0.93, 773), (0.97, 788)]


def _nearest_s(track, xy):
    return float(track.s[np.linalg.norm(track.center - np.asarray(xy), axis=1).argmin()])


def build_interlagos() -> Track:
    px = np.array(MAP_POINTS_PX, dtype=float)
    pts = np.stack([px[:, 0], -px[:, 1]], axis=1)          # y da imagem aponta para baixo
    scale = 1.0
    for _ in range(8):   # ajusta a escala (m/pixel) até a extensão ficar em 4.309 m
        track = Track.from_control_points("Interlagos", pts * scale, TRACK_WIDTH, smooth_m=25.0)
        if abs(track.length - LENGTH_M) < 0.05:
            break
        scale *= LENGTH_M / track.length
    track.set_elevation([f for f, _ in ELEVATION], [e for _, e in ELEVATION], drop=ELEVATION_DROP)
    track.elevation += ALTITUDE_AVG - track.elevation.mean()
    near = lambda q: _nearest_s(track, (q[0] * scale, -q[1] * scale))
    track.set_drs([(near(a), near(b)) for a, b in DRS_PX])
    track.meta = {"circuit": "Autódromo José Carlos Pace", "laps": RACE_LAPS,
                  "race_km": round(RACE_LAPS * track.length / 1000, 2), "elevation_drop_m": ELEVATION_DROP,
                  "altitude_avg_m": round(float(track.elevation.mean())), "drs_zones": 2,
                  "metros_por_pixel_do_mapa": round(scale, 3)}
    return track
