"""Persistência: configuração, histórico (CSV), checkpoints por geração e melhor genoma."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HISTORY_FIELDS = ["gen", "best_fit", "mean_fit", "median_fit", "std_fit", "best_progress",
                  "mean_progress", "n_finishers", "best_time", "sigma", "val_progress",
                  "val_finish_rate", "elapsed_s"]


class RunDir:
    def __init__(self, path):
        self.path = Path(path)
        self.ckpt_dir = self.path / "checkpoints"

    def create(self):
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        return self

    # ---- config
    def save_config(self, cfg: dict):
        (self.path / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    def load_config(self) -> dict:
        return json.loads((self.path / "config.json").read_text())

    # ---- checkpoints
    def ckpt_path(self, gen: int) -> Path:
        return self.ckpt_dir / f"gen_{gen:04d}.npz"

    def save_checkpoint(self, gen, genomes, fitness, extra: dict | None = None):
        """Guarda a população ordenada do melhor para o pior (índice 0 = melhor da geração)."""
        order = np.argsort(-np.asarray(fitness))
        np.savez_compressed(self.ckpt_path(gen), genomes=np.asarray(genomes, dtype=np.float64)[order],
                            fitness=np.asarray(fitness, dtype=np.float32)[order], gen=gen,
                            meta=json.dumps(extra or {}))

    def load_checkpoint(self, gen: int) -> dict:
        with np.load(self.ckpt_path(gen)) as z:
            return {"genomes": z["genomes"].astype(np.float64), "fitness": z["fitness"], "gen": int(z["gen"]),
                    "meta": json.loads(str(z["meta"]))}

    def generations(self) -> list[int]:
        return sorted(int(p.stem.split("_")[1]) for p in self.ckpt_dir.glob("gen_*.npz"))

    def resolve_gen(self, which) -> int:
        gens = self.generations()
        if not gens:
            raise FileNotFoundError(f"nenhum checkpoint em {self.ckpt_dir}")
        if which in ("last", "ultima", None):
            return gens[-1]
        if which == "best":
            h = self.load_history()
            if not h:
                return gens[-1]
            return int(max(h, key=lambda r: (r["best_fit"], r["gen"]))["gen"])
        g = int(which)
        if g not in gens:
            raise ValueError(f"geração {g} não existe (disponíveis: {gens[0]}..{gens[-1]})")
        return g

    def best_genome(self, which="best") -> tuple[int, np.ndarray]:
        g = self.resolve_gen(which)
        return g, self.load_checkpoint(g)["genomes"][0]

    # ---- histórico
    def append_history(self, row: dict):
        f = self.path / "history.csv"
        new = not f.exists()
        with f.open("a", newline="") as fh:
            w = csv.DictWriter(fh, HISTORY_FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in HISTORY_FIELDS})

    def load_history(self) -> list[dict]:
        f = self.path / "history.csv"
        if not f.exists():
            return []
        rows = []
        with f.open() as fh:
            for r in csv.DictReader(fh):
                rows.append({k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()})
                rows[-1]["gen"] = int(rows[-1]["gen"])
        return rows

    def truncate_history(self, last_gen: int):
        rows = [r for r in self.load_history() if r["gen"] <= last_gen]
        f = self.path / "history.csv"
        f.unlink(missing_ok=True)
        for r in rows:
            self.append_history(r)
