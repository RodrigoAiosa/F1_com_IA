"""Piloto automático simples (baseline): perseguição de ponto + velocidade pela curvatura à frente.

Usa a linha central da pista (informação "de mapa"), ao contrário da rede neural, que só vê
sensores. Serve como referência de tempo de volta e como verificação de que uma pista é pilotável.
"""
from __future__ import annotations

import numpy as np

from .track import wrap_angle


class Autopilot:
    def __init__(self, a_comfort: float = 13.0, decel: float = 11.0, preview: float = 220.0,
                 lookahead_min: float = 6.0, lookahead_gain: float = 0.45, speed_gain: float = 0.25):
        self.a_comfort, self.decel, self.preview = a_comfort, decel, preview
        self.l0, self.lg, self.kv = lookahead_min, lookahead_gain, speed_gain

    def act(self, sim, ai):
        tr, car = sim.track, sim.car
        idx, v = sim.idx[ai], sim.v[ai]
        # --- direção: persegue um ponto da linha central à frente
        ld = self.l0 + self.lg * v
        tidx = (idx + np.ceil(ld / tr.ds).astype(int)) % tr.M
        tx, ty = tr.center[tidx, 0] - sim.x[ai], tr.center[tidx, 1] - sim.y[ai]
        alpha = wrap_angle(np.arctan2(ty, tx) - sim.h[ai])
        dist = np.maximum(np.hypot(tx, ty), 1.0)
        kappa = 2.0 * np.sin(alpha) / dist
        steer = np.clip(kappa / car.kmax, -1.0, 1.0)
        # --- velocidade: a mais alta que ainda permite frear antes de cada curva à frente
        n = int(self.preview / tr.ds)
        j = (idx[:, None] + np.arange(0, n)[None, :]) % tr.M
        kk = np.maximum(np.abs(tr.curvature[j]), 1e-4)
        v_corner = np.sqrt(self.a_comfort / kk)
        d = np.arange(0, n)[None, :] * tr.ds
        v_allow = np.sqrt(v_corner ** 2 + 2 * self.decel * d)
        v_target = np.minimum(v_allow.min(axis=1), car.max_speed * 0.97)
        thr = np.clip(self.kv * (v_target - v), -1.0, 1.0)
        return steer, thr
