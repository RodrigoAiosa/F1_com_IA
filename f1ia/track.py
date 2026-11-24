"""Pista 2D: geometria da linha central, grade de ocupação e consultas vetorizadas.

Este módulo não depende de nenhuma biblioteca gráfica. Tudo o que a simulação precisa
(está na pista? qual a distância até a borda em cada raio? quanto da volta foi feito?)
é respondido com consultas O(1) em uma grade booleana pré-calculada.
"""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Normaliza ângulos para [-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def catmull_rom_closed(points, samples_per_segment: int = 80, alpha: float = 0.5) -> np.ndarray:
    """Spline Catmull-Rom centrípeta fechada que passa por todos os pontos de controle."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** alpha
        t = np.linspace(t1, t2, samples_per_segment, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def resample_closed(points, ds: float, *extras):
    """Reamostra uma poligonal fechada em passos uniformes de comprimento de arco."""
    pts = np.asarray(points, dtype=float)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    m = max(16, int(round(total / ds)))
    s_new = np.linspace(0.0, total, m, endpoint=False)
    res = np.stack([np.interp(s_new, s, closed[:, 0]), np.interp(s_new, s, closed[:, 1])], axis=1)
    ex = []
    for e in extras:
        e = np.asarray(e, dtype=float)
        ec = np.concatenate([e, e[:1]])
        ex.append(np.interp(s_new, s, ec))
    return (res, *ex) if ex else res


class Track:
    """Pista fechada definida por linha central + largura.

    Atributos principais (todos em metros / radianos):
      center (M,2), tangent (M,2), normal (M,2, aponta para a esquerda), curvature (M,)
      width (M,), s (M,), ds, length
    """

    def __init__(self, name: str, center, width=14.0, ds: float = 3.0, cell: float = 1.0,
                 car_radius: float = 1.2):
        center = np.asarray(center, dtype=float)
        width_arr = np.full(len(center), float(width)) if np.isscalar(width) else np.asarray(width, float)
        self.name = name
        self.ds = float(ds)
        self.cell = float(cell)
        self.car_radius = float(car_radius)
        self.center, self.width = resample_closed(center, ds, width_arr)
        self.M = len(self.center)
        # o passo real difere um pouco de ds para fechar a volta exatamente
        seg = np.linalg.norm(np.roll(self.center, -1, axis=0) - self.center, axis=1)
        self.length = float(seg.sum())
        self.ds = self.length / self.M
        self.s = np.arange(self.M) * self.ds
        d = np.roll(self.center, -1, axis=0) - np.roll(self.center, 1, axis=0)
        self.tangent = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.normal = np.stack([-self.tangent[:, 1], self.tangent[:, 0]], axis=1)
        self.heading = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        dth = wrap_angle(np.roll(self.heading, -1) - np.roll(self.heading, 1)) / (2 * self.ds)
        k = np.ones(5) / 5
        pad = np.concatenate([dth[-2:], dth, dth[:2]])
        self.curvature = np.convolve(pad, k, mode="valid")  # + = esquerda
        self.start_pos = self.center[0].copy()
        self.start_heading = float(self.heading[0])
        self._build_grid()

    # ------------------------------------------------------------------ construção
    @classmethod
    def from_control_points(cls, name, points, width=14.0, **kw):
        dense = catmull_rom_closed(points)
        if not np.isscalar(width):  # largura por ponto de controle -> interpola ao longo da spline
            w = np.asarray(width, float)
            n = len(points)
            w = np.repeat(w, 80)[: len(dense)] if len(w) == n else w
            width = w
        return cls(name, dense, width, **kw)

    @classmethod
    def from_csv(cls, name, path, width=14.0, **kw):
        """CSV com colunas x,y (e opcionalmente largura) em metros."""
        data = np.loadtxt(path, delimiter=",", ndmin=2)
        w = data[:, 2] if data.shape[1] > 2 else width
        return cls(name, data[:, :2], w, **kw)

    def _build_grid(self):
        pad = 30.0
        half = self.width.max() / 2
        lo = self.center.min(axis=0) - half - pad
        hi = self.center.max(axis=0) + half + pad
        self.x0, self.y0 = float(lo[0]), float(lo[1])
        self.W = int(np.ceil((hi[0] - lo[0]) / self.cell)) + 1
        self.H = int(np.ceil((hi[1] - lo[1]) / self.cell)) + 1
        grid = np.zeros((self.H, self.W), dtype=bool)
        dense, wd = resample_closed(self.center, 1.0, self.width)
        c = self.cell
        for (px, py), w in zip(dense, wd):
            r = w / 2 - self.car_radius
            ix0 = max(int((px - r - self.x0) / c), 0)
            ix1 = min(int((px + r - self.x0) / c) + 1, self.W - 1)
            iy0 = max(int((py - r - self.y0) / c), 0)
            iy1 = min(int((py + r - self.y0) / c) + 1, self.H - 1)
            xs = self.x0 + (np.arange(ix0, ix1 + 1) + 0.5) * c
            ys = self.y0 + (np.arange(iy0, iy1 + 1) + 0.5) * c
            m = (xs[None, :] - px) ** 2 + (ys[:, None] - py) ** 2 <= r * r
            grid[iy0:iy1 + 1, ix0:ix1 + 1] |= m
        self.grid = grid

    # ------------------------------------------------------------------ consultas
    def on_track(self, x, y):
        ix = np.floor((np.asarray(x) - self.x0) / self.cell).astype(np.intp)
        iy = np.floor((np.asarray(y) - self.y0) / self.cell).astype(np.intp)
        inside = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        val = self.grid[np.clip(iy, 0, self.H - 1), np.clip(ix, 0, self.W - 1)]
        return val & inside

    def raycast(self, x, y, angles, max_range: float = 100.0, step: float = 2.0):
        """Distância até a borda da pista para cada raio. x,y: (k,), angles: (k,R) -> (k,R)."""
        steps = np.arange(step, max_range + 1e-9, step)
        dx, dy = np.cos(angles), np.sin(angles)
        px = x[:, None, None] + dx[:, :, None] * steps[None, None, :]
        py = y[:, None, None] + dy[:, :, None] * steps[None, None, :]
        off = ~self.on_track(px, py)
        first = off.argmax(axis=2)
        hit = off.any(axis=2)
        return np.where(hit, steps[first], max_range)

    def locate(self, x, y, idx_prev, back: int = 8, fwd: int = 20):
        """Posição ao longo da pista, buscando só perto do índice anterior.

        Retorna (idx, s_local, lateral): índice mais próximo, arco (m) com interpolação
        e deslocamento lateral assinado (+ = esquerda da linha central).
        """
        cand = (idx_prev[:, None] + np.arange(-back, fwd + 1)[None, :]) % self.M
        c = self.center[cand]
        d2 = (c[..., 0] - x[:, None]) ** 2 + (c[..., 1] - y[:, None]) ** 2
        j = d2.argmin(axis=1)
        idx = cand[np.arange(len(idx_prev)), j]
        rx, ry = x - self.center[idx, 0], y - self.center[idx, 1]
        along = rx * self.tangent[idx, 0] + ry * self.tangent[idx, 1]
        lateral = rx * self.normal[idx, 0] + ry * self.normal[idx, 1]
        return idx, self.s[idx] + along, lateral

    # ------------------------------------------------------------------ exportação
    @property
    def left(self):
        return self.center + self.normal * (self.width / 2)[:, None]

    @property
    def right(self):
        return self.center - self.normal * (self.width / 2)[:, None]

    def bounds(self):
        pts = np.vstack([self.left, self.right])
        return pts.min(axis=0), pts.max(axis=0)

    def to_dict(self, precision: int = 1) -> dict:
        r = lambda a: np.round(a, precision).tolist()
        return {
            "name": self.name,
            "length": round(self.length, 1),
            "width": round(float(self.width.mean()), 1),
            "start": {"x": round(float(self.start_pos[0]), 2), "y": round(float(self.start_pos[1]), 2),
                      "heading": round(self.start_heading, 4)},
            "center": r(self.center),
            "left": r(self.left),
            "right": r(self.right),
        }
