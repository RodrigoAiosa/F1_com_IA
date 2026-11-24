"""Simulação 100% headless: física do carro, sensores, progresso e condições de término.

Nada aqui importa pygame ou matplotlib. A interface gráfica apenas lê o estado (x, y, heading...)
deste objeto, então o mesmo código roda no treino em lote e na janela.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .track import Track, wrap_angle

# códigos de término
ALIVE, OFF_TRACK, STALLED, WRONG_WAY, FINISHED, TIMEOUT = 0, 1, 2, 3, 4, 5
REASON_NAMES = {ALIVE: "em corrida", OFF_TRACK: "saiu da pista", STALLED: "parado",
                WRONG_WAY: "contramão", FINISHED: "completou", TIMEOUT: "tempo esgotado"}


@dataclass(frozen=True)
class CarParams:
    wheelbase: float = 2.8
    max_steer: float = 0.45      # rad
    max_speed: float = 45.0      # m/s
    accel: float = 14.0          # m/s²
    brake: float = 24.0          # m/s²
    a_lat: float = 22.0          # aderência lateral máxima (m/s²); acima disso o carro "abre" a trajetória
    steer_rate: float = 5.0      # 1/s (volante vai de -1 a 1 em ~0,4 s)
    drs_boost: float = 0.06      # +6% de velocidade máxima nas zonas de DRS
    drag: float = 0.05           # 1/s: arrasto proporcional à velocidade (carro solta e desacelera)
    length: float = 4.6          # só para desenho
    width: float = 1.9

    @property
    def kmax(self) -> float:
        return float(np.tan(self.max_steer) / self.wheelbase)


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.05
    n_laps: int = 1
    max_time: float | None = None    # None -> length * n_laps / min_avg_speed
    min_avg_speed: float = 18.0      # m/s: abaixo disso não dá tempo de completar a volta
    stall_time: float = 8.0
    stall_dist: float = 4.0          # precisa avançar pelo menos isto a cada stall_time
    wrong_way: float = 30.0
    ray_angles_deg: tuple = (-90.0, -50.0, -25.0, 0.0, 25.0, 50.0, 90.0)
    ray_range: float = 100.0
    ray_step: float = 2.0

    @property
    def n_inputs(self) -> int:
        return len(self.ray_angles_deg) + 2   # raios + velocidade + volante


class Sim:
    """N carros independentes (sem colisão entre eles) na mesma pista."""

    def __init__(self, track: Track, n_cars: int = 1, car: CarParams = CarParams(),
                 cfg: SimConfig = SimConfig(), record_stride: int = 0):
        self.track, self.car, self.cfg, self.n = track, car, cfg, int(n_cars)
        self.ray_angles = np.deg2rad(np.asarray(cfg.ray_angles_deg))
        self.max_time = cfg.max_time or track.length * cfg.n_laps / cfg.min_avg_speed
        self.max_steps = int(self.max_time / cfg.dt)
        self.record_stride = record_stride
        self.reset()

    # ------------------------------------------------------------------ estado
    def reset(self):
        n, t = self.n, self.track
        self.step_count = 0
        self.x = np.full(n, t.start_pos[0])
        self.y = np.full(n, t.start_pos[1])
        self.h = np.full(n, t.start_heading)
        self.v = np.zeros(n)
        self.steer = np.zeros(n)
        self.idx = np.zeros(n, dtype=np.intp)
        self.s_prev = np.zeros(n)
        self.progress = np.zeros(n)          # metros acumulados (pode ser negativo)
        self.best = np.zeros(n)              # maior progresso já atingido
        self.alive = np.ones(n, dtype=bool)
        self.reason = np.zeros(n, dtype=np.int8)
        self.finish_time = np.full(n, np.nan)
        self.last_gain_step = np.zeros(n, dtype=np.int64)
        self.last_gain_val = np.zeros(n)
        self.dist = np.zeros(n)
        self.lateral = np.zeros(n)
        self.frames: list[np.ndarray] = []
        self._record()

    @property
    def alive_idx(self) -> np.ndarray:
        return np.flatnonzero(self.alive)

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())

    @property
    def time(self) -> float:
        return self.step_count * self.cfg.dt

    @property
    def total_distance(self) -> float:
        return self.track.length * self.cfg.n_laps

    # ------------------------------------------------------------------ sensores
    def ray_distances(self, ai) -> np.ndarray:
        ang = self.h[ai][:, None] + self.ray_angles[None, :]
        return self.track.raycast(self.x[ai], self.y[ai], ang, self.cfg.ray_range, self.cfg.ray_step)

    def observe(self, ai) -> np.ndarray:
        rays = self.ray_distances(ai) / self.cfg.ray_range
        return np.concatenate([rays, (self.v[ai] / self.car.max_speed)[:, None],
                               self.steer[ai][:, None]], axis=1)

    # ------------------------------------------------------------------ dinâmica
    def step(self, ai, steer_cmd, throttle_cmd):
        """Avança um passo os carros em `ai` (índices vivos) com comandos em [-1, 1]."""
        if len(ai) == 0:
            return
        c, cfg = self.car, self.cfg
        dt = cfg.dt
        st = self.steer[ai]
        cmd = np.clip(steer_cmd, -1.0, 1.0)
        st = st + np.clip(cmd - st, -c.steer_rate * dt, c.steer_rate * dt)
        self.steer[ai] = st
        thr = np.clip(throttle_cmd, -1.0, 1.0)
        v = self.v[ai]
        cur = self.idx[ai]
        vmax = c.max_speed * (1.0 + c.drs_boost * self.track.drs[cur])   # DRS: mais velocidade final nas retas
        acc = np.where(thr >= 0, c.accel * thr * np.maximum(0.0, 1.0 - v / vmax), c.brake * thr) - c.drag * v
        acc = acc - 9.81 * self.track.slope[cur]                         # subida freia, descida acelera
        v = np.clip(v + acc * dt, 0.0, vmax)
        kappa = st * c.kmax
        klim = c.a_lat / np.maximum(v, 1.0) ** 2
        kappa = np.clip(kappa, -klim, klim)
        h0 = self.h[ai]
        h1 = h0 + v * kappa * dt
        hm = 0.5 * (h0 + h1)
        x = self.x[ai] + v * np.cos(hm) * dt
        y = self.y[ai] + v * np.sin(hm) * dt
        self.x[ai], self.y[ai], self.h[ai], self.v[ai] = x, y, h1, v
        self.dist[ai] += v * dt
        self.step_count += 1

        tr = self.track
        idx, s_loc, lat = tr.locate(x, y, self.idx[ai])
        delta = wrap_angle((s_loc - self.s_prev[ai]) * (2 * np.pi / tr.length)) * (tr.length / (2 * np.pi))
        prog = self.progress[ai] + delta
        self.idx[ai], self.s_prev[ai], self.lateral[ai], self.progress[ai] = idx, s_loc, lat, prog
        best = np.maximum(self.best[ai], prog)
        self.best[ai] = best

        # condições de término
        off = ~tr.on_track(x, y)
        gained = best - self.last_gain_val[ai] >= cfg.stall_dist
        self.last_gain_step[ai] = np.where(gained, self.step_count, self.last_gain_step[ai])
        self.last_gain_val[ai] = np.where(gained, best, self.last_gain_val[ai])
        stalled = (self.step_count - self.last_gain_step[ai]) * dt > cfg.stall_time
        wrong = prog < best - cfg.wrong_way
        done = best >= self.total_distance

        reason = np.zeros(len(ai), dtype=np.int8)
        reason[stalled] = STALLED
        reason[wrong] = WRONG_WAY
        reason[off] = OFF_TRACK
        reason[done & ~off] = FINISHED
        newly = reason != ALIVE
        if newly.any():
            who = ai[newly]
            self.alive[who] = False
            self.reason[who] = reason[newly]
            fin = who[reason[newly] == FINISHED]
            self.finish_time[fin] = self.time
        if self.step_count >= self.max_steps:
            left = self.alive_idx
            self.alive[left] = False
            self.reason[left] = TIMEOUT
        self._record()

    def _record(self):
        if self.record_stride and self.step_count % self.record_stride == 0:
            self.frames.append(np.stack([self.x, self.y, self.h, self.alive.astype(float)], axis=1))

    # ------------------------------------------------------------------ execução
    def run(self, policy) -> "Sim":
        """Roda até todos os carros terminarem."""
        while self.n_alive > 0:
            ai = self.alive_idx
            steer, thr = policy.act(self, ai)
            self.step(ai, steer, thr)
        return self

    # ------------------------------------------------------------------ resultados
    def result(self) -> dict:
        total = self.total_distance
        frac = np.clip(self.best / total, 0.0, 1.0)
        finished = self.reason == FINISHED
        t = np.where(finished, self.finish_time, self.max_time)
        bonus = np.where(finished, 0.25 + 0.5 * np.clip(1.0 - t / self.max_time, 0.0, 1.0), 0.0)
        return {
            "fitness": frac + bonus,
            "progress": frac,
            "finished": finished,
            "time": np.where(finished, self.finish_time, np.nan),
            "reason": self.reason.copy(),
            "avg_speed": self.dist / np.maximum(self.step_count * self.cfg.dt, 1e-9),
        }

    def trajectory(self) -> np.ndarray:
        """(T, n, 4): x, y, heading, vivo — requer record_stride > 0."""
        return np.stack(self.frames) if self.frames else np.zeros((0, self.n, 4))
