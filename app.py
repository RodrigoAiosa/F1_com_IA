"""API Python para a Vercel (FastAPI). O visualizador estático fica em public/.

Rotas:
  GET /api/health
  GET /api/models                                  modelos treinados disponíveis (pasta models/)
  GET /api/track?track=interlagos|seed:<n>         geometria da pista
  GET /api/simulate?track=...&model=autopilot|<nome>   roda 1 carro e devolve a trajetória
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query

from f1ia.autopilot import Autopilot
from f1ia.nn import PopulationPolicy, Topology
from f1ia.sim import REASON_NAMES, CarParams, Sim, SimConfig
from f1ia.tracks import get_track

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
TRACK_RE = re.compile(r"^(interlagos|seed:\d{1,7})$")
MODEL_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
RECORD_STRIDE = 2

app = FastAPI(title="F1 com IA", description="Neuroevolução de carros de corrida")


def _track(spec: str):
    if not TRACK_RE.match(spec):
        raise HTTPException(400, "track deve ser 'interlagos' ou 'seed:<número>'")
    return get_track(spec)


@lru_cache(maxsize=8)
def _load_model(name: str):
    path = MODELS_DIR / f"{name}.json"
    if not MODEL_RE.match(name) or not path.is_file():
        raise HTTPException(404, f"modelo '{name}' não encontrado")
    d = json.loads(path.read_text())
    topo = Topology.from_dict(d["topology"])
    cfg = SimConfig(**{**d["sim"], "ray_angles_deg": tuple(d["sim"]["ray_angles_deg"])})
    return d, topo, cfg, np.asarray(d["genome"], dtype=float)[None, :]


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/models")
def models():
    out = []
    for p in sorted(MODELS_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        out.append({"name": d["name"], "generation": d.get("generation"), "params": len(d["genome"])})
    return {"models": out}


@app.get("/api/track")
def track(track: str = Query("interlagos")):
    return _track(track).to_dict()


@app.get("/api/simulate")
def simulate(track: str = Query("interlagos"), model: str = Query("autopilot")):
    tr = _track(track)
    car = CarParams()
    if model == "autopilot":
        cfg = SimConfig()
        sim = Sim(tr, 1, car, cfg, record_stride=RECORD_STRIDE)
        sim.run(Autopilot())
        label = "Piloto automático"
    else:
        d, topo, cfg, genome = _load_model(model)
        sim = Sim(tr, 1, car, cfg, record_stride=RECORD_STRIDE)
        sim.run(PopulationPolicy(genome, topo))
        label = f"{d['name']} (ger {d.get('generation')})"
    r = {k: v[0] for k, v in sim.result().items()}
    traj = sim.trajectory()[:, 0, :]
    return {
        "track": tr.to_dict(),
        "car": {"label": label, "dt": cfg.dt * RECORD_STRIDE,
                "frames": [[round(float(x), 1), round(float(y), 1), round(float(h), 2)] for x, y, h, _ in traj]},
        "result": {"progress": float(r["progress"]), "finished": bool(r["finished"]),
                   "time": None if np.isnan(r["time"]) else float(r["time"]),
                   "reason": REASON_NAMES[int(r["reason"])], "avg_speed": float(r["avg_speed"])},
    }
