"""Treino em lote (sem janela): avalia, registra, salva checkpoint e evolui, geração a geração."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .evaluate import Evaluator, benchmark
from .evolution import EvoConfig, Evolution
from .nn import Topology
from .sim import CarParams, SimConfig
from .storage import RunDir
from .tracks import parse_specs


@dataclass
class TrainConfig:
    run: str = "runs/interlagos"
    tracks: list = field(default_factory=lambda: ["interlagos"])
    seed: int = 0
    generations: int = 100
    hidden: tuple = (12, 8)
    workers: int = 1
    val_tracks: list = field(default_factory=lambda: [f"seed:{i}" for i in range(1000, 1006)])
    val_every: int = 5
    checkpoint_every: int = 1
    overwrite: bool = False
    stop_when_finishers: float | None = None   # ex.: 0.5 -> para quando 50% da população completa
    evo: EvoConfig = field(default_factory=EvoConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        d["sim"]["ray_angles_deg"] = list(self.sim.ray_angles_deg)
        return d


class Trainer:
    def __init__(self, cfg: TrainConfig, resume: bool = False):
        self.cfg = cfg
        self.topology = Topology(cfg.sim.n_inputs, tuple(cfg.hidden), 2)
        self.car = CarParams()
        self.rd = RunDir(cfg.run).create()
        self.evo = Evolution(self.topology, cfg.evo, cfg.seed)
        self.evaluator = Evaluator(self.topology, cfg.tracks, cfg.sim, self.car, cfg.workers)
        self.t_start = time.time()
        self.t_offset = 0.0
        if resume and self.rd.generations():
            last = self.rd.generations()[-1]
            ck = self.rd.load_checkpoint(last)
            self.rd.truncate_history(last)
            # a população salva é a que FOI avaliada em `last`; reconstruímos a próxima a partir dela
            self.evo.population = ck["genomes"]
            self.evo.generation = last - 1
            self.evo.best_fitness = float(max((r["best_fit"] for r in self.rd.load_history()), default=-np.inf))
            self.evo.next_generation(ck["fitness"])
            h = self.rd.load_history()
            self.t_offset = h[-1]["elapsed_s"] if h else 0.0
        else:
            if self.rd.generations() or (self.rd.path / "history.csv").exists():
                if not cfg.overwrite:
                    raise FileExistsError(f"{cfg.run} já tem um treino. Use --resume, --overwrite ou outro --run.")
                for f in self.rd.ckpt_dir.glob("gen_*.npz"):
                    f.unlink()
                (self.rd.path / "history.csv").unlink(missing_ok=True)
            self.rd.save_config(cfg.to_dict())

    @property
    def generation(self) -> int:
        """Número (a partir de 1) da geração que será avaliada em seguida."""
        return self.evo.generation + 1

    # ------------------------------------------------------------------
    def complete_generation(self, per_track: list[dict], val: dict | None = None) -> dict:
        """Registra uma geração já avaliada (`per_track`: um dict por pista) e evolui a população."""
        cfg = self.cfg
        gen = self.generation
        pop = self.evo.population
        fit = np.mean([r["fitness"] for r in per_track], axis=0)
        prog = np.mean([r["progress"] for r in per_track], axis=0)
        fin_all = np.all([r["finished"] for r in per_track], axis=0)
        b = int(fit.argmax())
        t0 = per_track[0]["time"]
        best_time = float(np.nanmin(t0)) if np.isfinite(t0).any() else float("nan")
        row = {"gen": gen, "best_fit": float(fit[b]), "mean_fit": float(fit.mean()),
               "median_fit": float(np.median(fit)), "std_fit": float(fit.std()),
               "best_progress": float(prog[b]), "mean_progress": float(prog.mean()),
               "n_finishers": int(fin_all.sum()), "best_time": best_time, "sigma": self.evo.sigma,
               "elapsed_s": self.t_offset + time.time() - self.t_start}
        if val:
            row["val_progress"], row["val_finish_rate"] = val["mean_progress"], val["finish_rate"]
        if gen % cfg.checkpoint_every == 0 or gen == 1:
            self.rd.save_checkpoint(gen, pop, fit, {"best_progress": row["best_progress"]})
        self.rd.append_history(row)
        self.evo.next_generation(fit)
        return row

    def validate(self, genome) -> dict:
        _, summary = benchmark(genome, self.topology, self.cfg.val_tracks, self.cfg.sim, self.car)
        return summary

    def run(self, n_gens: int | None = None, callback=None):
        cfg = self.cfg
        end = self.generation + (n_gens or cfg.generations) - 1
        rows = []
        try:
            while self.generation <= end:
                gen = self.generation
                per_track = self.evaluator.evaluate(self.evo.population)
                fit = np.mean([r["fitness"] for r in per_track], axis=0)
                val = None
                if cfg.val_tracks and cfg.val_every and (gen % cfg.val_every == 0 or gen == 1):
                    val = self.validate(self.evo.population[int(fit.argmax())])
                row = self.complete_generation(per_track, val)
                rows.append(row)
                if callback:
                    callback(row)
                if cfg.stop_when_finishers and row["n_finishers"] >= cfg.stop_when_finishers * cfg.evo.pop_size:
                    break
        finally:
            self.evaluator.close()
        return rows


def format_row(r: dict) -> str:
    t = "  --  " if np.isnan(r["best_time"]) else f"{r['best_time']:6.1f}s"
    v = "" if np.isnan(r.get("val_progress", float("nan"))) else \
        f" | val {r['val_progress'] * 100:5.1f}% ({r['val_finish_rate'] * 100:3.0f}% ok)"
    return (f"ger {r['gen']:4d} | melhor {r['best_fit']:.3f} média {r['mean_fit']:.3f} | "
            f"prog {r['best_progress'] * 100:5.1f}% | completaram {int(r['n_finishers']):3d} | volta {t} | "
            f"σ {r['sigma']:.3f} | {r['elapsed_s']:6.1f}s{v}")
