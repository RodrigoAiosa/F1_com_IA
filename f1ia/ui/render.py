"""Desenho: pista pré-renderizada, câmera, carros, sensores e HUD."""
from __future__ import annotations

import math

import numpy as np
import pygame

from ..track import resample_closed

BG = (34, 52, 38)
ASPHALT = (62, 64, 70)
EDGE = (235, 235, 235)
KERB_A, KERB_B = (200, 40, 40), (240, 240, 240)
TEXT = (235, 235, 235)
PRERENDER_SCALE = 2.0  # px por metro


class TrackSurface:
    """Pista desenhada uma única vez em uma Surface grande (coordenadas de mundo -> pixels)."""

    def __init__(self, track, scale: float = PRERENDER_SCALE):
        self.track, self.scale = track, scale
        lo, hi = track.bounds()
        pad = 40.0
        self.x0, self.y1 = lo[0] - pad, hi[1] + pad
        w = int((hi[0] - lo[0] + 2 * pad) * scale)
        h = int((hi[1] - lo[1] + 2 * pad) * scale)
        self.world_w, self.world_h = w / scale, h / scale
        self.cx, self.cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        surf = pygame.Surface((w, h))
        surf.fill(BG)
        dense, wd = resample_closed(track.center, 1.5, track.width)
        for (x, y), ww in zip(dense, wd):
            pygame.draw.circle(surf, ASPHALT, self._p(x, y), int(ww / 2 * scale))
        for edge in (track.left, track.right):
            pygame.draw.lines(surf, EDGE, True, [self._p(x, y) for x, y in edge], max(2, int(0.35 * scale)))
        # linha de largada xadrez
        c, n, wdt = track.center[0], track.normal[0], track.width[0]
        t = track.tangent[0]
        for i in range(8):
            a = -wdt / 2 + wdt * i / 8
            b = a + wdt / 8
            for j in range(2):
                col = (255, 255, 255) if (i + j) % 2 == 0 else (20, 20, 20)
                p0 = c + n * a + t * (j * 1.2)
                p1 = c + n * b + t * (j * 1.2)
                pygame.draw.polygon(surf, col, [self._p(*p0), self._p(*p1), self._p(*(p1 + t * 1.2)), self._p(*(p0 + t * 1.2))])
        self.surface = surf
        self._cache: dict = {}

    def _p(self, x, y):
        return int((x - self.x0) * self.scale), int((self.y1 - y) * self.scale)


class Camera:
    def __init__(self, ts: TrackSurface, size):
        self.ts, self.W, self.H = ts, size[0], size[1]
        self.fit = False
        self.zoom_follow = 2.0
        self.cx, self.cy = ts.cx, ts.cy

    @property
    def fit_zoom(self):
        return min(self.W / self.ts.world_w, self.H / self.ts.world_h)

    @property
    def zoom(self):
        return self.fit_zoom if self.fit else self.zoom_follow

    def toggle(self):
        self.fit = not self.fit

    def zoom_by(self, f):
        self.zoom_follow = float(np.clip(self.zoom_follow * f, 0.4, 6.0))

    def look_at(self, x, y):
        if self.fit:
            self.cx, self.cy = self.ts.cx, self.ts.cy
        else:
            self.cx, self.cy = x, y

    def to_screen(self, x, y):
        z = self.zoom
        return (self.W / 2 + (x - self.cx) * z, self.H / 2 - (y - self.cy) * z)

    def draw_track(self, screen):
        ts, z = self.ts, self.zoom
        S = ts.scale
        screen.fill(BG)
        sw, sh = self.W / z * S, self.H / z * S
        csx, csy = (self.cx - ts.x0) * S, (ts.y1 - self.cy) * S
        rect = pygame.Rect(int(csx - sw / 2), int(csy - sh / 2), int(sw) + 2, int(sh) + 2)
        full = ts.surface.get_rect()
        clip = rect.clip(full)
        if clip.w <= 0 or clip.h <= 0:
            return
        k = z / S
        dest = (int((clip.x - rect.x) * k), int((clip.y - rect.y) * k))
        if clip == full:
            key = round(z, 3)
            if key not in ts._cache:
                ts._cache.clear()
                ts._cache[key] = pygame.transform.smoothscale(ts.surface, (max(1, int(full.w * k)), max(1, int(full.h * k))))
            screen.blit(ts._cache[key], dest)
            return
        sub = ts.surface.subsurface(clip)
        if abs(k - 1.0) > 1e-3:
            sub = pygame.transform.smoothscale(sub, (max(1, int(clip.w * k)), max(1, int(clip.h * k))))
        screen.blit(sub, dest)


def car_polygon(cam: Camera, x, y, h, length=4.6, width=1.9, min_px=9.0):
    z = cam.zoom
    f = max(1.0, min_px / (length * z))
    l, w = length * f / 2, width * f / 2
    c, s = math.cos(h), math.sin(h)
    pts = []
    for lx, ly in ((l, w), (l, -w), (-l, -w), (-l, w)):
        pts.append(cam.to_screen(x + lx * c - ly * s, y + lx * s + ly * c))
    return pts


def draw_car(screen, cam, x, y, h, color, outline=(10, 10, 10), alpha=None, car=None):
    L, W = (car.length, car.width) if car else (4.6, 1.9)
    pts = car_polygon(cam, x, y, h, L, W)
    if alpha is not None:
        surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(surf, (*color, alpha), pts)
        screen.blit(surf, (0, 0))
        return
    pygame.draw.polygon(screen, color, pts)
    pygame.draw.polygon(screen, outline, pts, 1)
    nose = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
    pygame.draw.circle(screen, (255, 255, 255), nose, 2)


def draw_rays(screen, cam, sim, i=0):
    ai = np.array([i])
    d = sim.ray_distances(ai)[0]
    ang = sim.h[i] + sim.ray_angles
    p0 = cam.to_screen(sim.x[i], sim.y[i])
    for dist, a in zip(d, ang):
        p1 = cam.to_screen(sim.x[i] + dist * math.cos(a), sim.y[i] + dist * math.sin(a))
        frac = dist / sim.cfg.ray_range
        pygame.draw.line(screen, (int(255 * (1 - frac)), int(220 * frac + 35), 60), p0, p1, 1)
        pygame.draw.circle(screen, (255, 200, 0), p1, 2)


class Hud:
    def __init__(self):
        self.font = pygame.font.Font(None, 24)
        self.big = pygame.font.Font(None, 44)
        self.small = pygame.font.Font(None, 19)

    def text(self, screen, s, pos, color=TEXT, font=None, shadow=True):
        f = font or self.font
        if shadow:
            screen.blit(f.render(s, True, (0, 0, 0)), (pos[0] + 1, pos[1] + 1))
        screen.blit(f.render(s, True, color), pos)

    def lines(self, screen, items, x=12, y=10, dy=22):
        for i, s in enumerate(items):
            self.text(screen, s, (x, y + i * dy))

    def center_msg(self, screen, s, sub=None):
        w, h = screen.get_size()
        t = self.big.render(s, True, (255, 255, 255))
        pygame.draw.rect(screen, (0, 0, 0), t.get_rect(center=(w // 2, h // 2)).inflate(40, 30))
        screen.blit(t, t.get_rect(center=(w // 2, h // 2)))
        if sub:
            self.text(screen, sub, (w // 2 - self.font.size(sub)[0] // 2, h // 2 + 34))

    def chart(self, screen, rect, series, title=""):
        """series: lista de (valores, cor). Eixo x = índice da geração."""
        pygame.draw.rect(screen, (15, 18, 24), rect)
        pygame.draw.rect(screen, (90, 90, 100), rect, 1)
        self.text(screen, title, (rect.x + 6, rect.y + 4), font=self.small, shadow=False)
        vals = [v for s, _ in series for v in s if np.isfinite(v)]
        if len(vals) < 2:
            return
        lo, hi = min(vals), max(vals)
        hi = hi if hi > lo else lo + 1
        n = max(len(s) for s, _ in series)
        for s, color in series:
            pts = [(rect.x + 6 + (rect.w - 12) * i / max(n - 1, 1),
                    rect.bottom - 6 - (rect.h - 28) * (v - lo) / (hi - lo)) for i, v in enumerate(s) if np.isfinite(v)]
            if len(pts) > 1:
                pygame.draw.lines(screen, color, False, pts, 2)
