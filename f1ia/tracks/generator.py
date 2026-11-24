"""Gerador de pistas por semente: curva estrelada com harmônicos aleatórios + validação.

A mesma semente sempre produz a mesma pista. A pista é rejeitada e re-sorteada (de forma
determinística a partir da semente) se tiver curvas fechadas demais ou trechos que se
aproximam a ponto de as bordas se tocarem.
"""
from __future__ import annotations

import numpy as np

from ..track import Track, catmull_rom_closed, resample_closed

MIN_RADIUS = 30.0


def _candidate(rng: np.random.Generator):
    n = 360
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones(n)
    for k in range(2, 9):
        amp = rng.uniform(0.0, 1.0) * (0.34 / k ** 0.75)
        r += amp * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
    if r.min() < 0.4:
        return None
    aspect = rng.uniform(0.55, 1.0)
    pts = np.stack([r * np.cos(theta), aspect * r * np.sin(theta)], axis=1)
    target = rng.uniform(3200, 5000)
    length = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum()
    return pts * (target / length), float(rng.uniform(13.0, 17.0))


def _valid(track: Track) -> bool:
    if 1.0 / np.abs(track.curvature).max() < MIN_RADIUS:
        return False
    # bordas não podem se aproximar de trechos distantes ao longo da pista
    c = track.center
    sub = c[:: max(1, track.M // 600)]
    sidx = np.arange(0, track.M, max(1, track.M // 600))
    d = np.linalg.norm(sub[:, None] - sub[None], axis=2)
    arc = np.abs(sidx[:, None] - sidx[None]) * track.ds
    arc = np.minimum(arc, track.length - arc)
    far = arc > 4 * track.width.max()
    return bool(np.all(d[far] > track.width.max() + 10.0))


def generate_track(seed: int, max_attempts: int = 200) -> Track:
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        cand = _candidate(rng)
        if cand is None:
            continue
        pts, width = cand
        center = resample_closed(catmull_rom_closed(pts, 4), 3.0)
        track = Track(f"seed:{seed}", center, width)
        if _valid(track):
            return track
    # fallback: elipse suave (sempre válida)
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    return Track(f"seed:{seed}", np.stack([700 * np.cos(t), 400 * np.sin(t)], axis=1), 15.0)
