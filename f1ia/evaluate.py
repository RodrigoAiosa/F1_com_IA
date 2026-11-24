"""Avaliação de genomas em uma ou várias pistas (com opção de paralelizar em processos)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .nn import PopulationPolicy, Topology
from .sim import CarParams, Sim, SimConfig
from .tracks import get_track


def _eval_chunk(args):
    genomes, topo_d, spec, sim_cfg, car = args
    topo = Topology.from_dict(topo_d)
    sim = Sim(get_track(spec), len(genomes), car, sim_cfg)
    sim.run(PopulationPolicy(genomes, topo))
    return sim.result()


def _merge(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class Evaluator:
    """Avalia uma população em todas as pistas. `workers>1` divide a população em processos."""

    def __init__(self, topology: Topology, track_specs, sim_cfg: SimConfig = SimConfig(),
                 car: CarParams = CarParams(), workers: int = 1):
        self.topology, self.specs, self.sim_cfg, self.car = topology, list(track_specs), sim_cfg, car
        self.workers = max(1, int(workers))
        self._pool = ProcessPoolExecutor(self.workers) if self.workers > 1 else None

    def close(self):
        if self._pool:
            self._pool.shutdown()

    def evaluate(self, genomes: np.ndarray, specs=None) -> list[dict]:
        """Retorna uma lista (uma por pista) de dicts com arrays de tamanho P."""
        specs = list(specs or self.specs)
        td = self.topology.to_dict()
        jobs = []
        for spec in specs:
            chunks = np.array_split(np.arange(len(genomes)), self.workers) if self._pool else [np.arange(len(genomes))]
            jobs.append([(genomes[c], td, spec, self.sim_cfg, self.car) for c in chunks if len(c)])
        if self._pool:
            flat = [j for job in jobs for j in job]
            res = list(self._pool.map(_eval_chunk, flat))
            out, i = [], 0
            for job in jobs:
                out.append(_merge(res[i:i + len(job)]))
                i += len(job)
            return out
        return [_merge([_eval_chunk(j) for j in job]) for job in jobs]


def run_genome(genome, topology: Topology, spec: str, sim_cfg: SimConfig = SimConfig(),
               car: CarParams = CarParams(), record_stride: int = 0):
    """Roda um único genoma e devolve (sim, resultado)."""
    sim = Sim(get_track(spec), 1, car, sim_cfg, record_stride=record_stride)
    sim.run(PopulationPolicy(np.asarray(genome)[None, :], topology))
    return sim, {k: v[0] for k, v in sim.result().items()}


def benchmark(genome, topology: Topology, specs, sim_cfg: SimConfig = SimConfig(), car: CarParams = CarParams()):
    """Avalia um genoma em várias pistas; devolve linhas por pista e um resumo."""
    rows = []
    for spec in specs:
        sim, r = run_genome(genome, topology, spec, sim_cfg, car)
        rows.append({"track": spec, "progress": float(r["progress"]), "finished": bool(r["finished"]),
                     "time": None if np.isnan(r["time"]) else float(r["time"]), "reason": int(r["reason"]),
                     "length": float(sim.track.length)})
    summary = {"n": len(rows), "finish_rate": float(np.mean([r["finished"] for r in rows])),
               "mean_progress": float(np.mean([r["progress"] for r in rows]))}
    return rows, summary
