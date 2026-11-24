"""Exporta replays, histórico e relatório de generalização como JSON para o visualizador web."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .compare import compare_generations, load_run
from .evaluate import benchmark
from .sim import CarParams, REASON_NAMES


def traj_to_json(traj: np.ndarray, label: str, dt: float, extra: dict | None = None) -> dict:
    d = {"label": label, "dt": dt,
         "frames": [[round(float(x), 1), round(float(y), 1), round(float(h), 2)] for x, y, h in traj[:, :3]]}
    d.update(extra or {})
    return d


def export_web(run_path, out_dir, track_spec="interlagos", gens=(1, 5, 10, "best"), gen_model="best",
               generalization_specs=None, model_dir="models", model_name="interlagos_best"):
    out, models = Path(out_dir), Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    rd, topo, sim_cfg = load_run(run_path)
    stride = 2
    dt = sim_cfg.dt * stride
    track, items = compare_generations(run_path, gens, track_spec, record_stride=stride)
    (out / "replays.json").write_text(json.dumps({
        "track": track.to_dict(),
        "cars": [traj_to_json(it["traj"], it["label"], dt, {"gen": it["gen"], "progress": it["row"]["progress"],
                                                           "finished": it["row"]["finished"],
                                                           "time": it["row"]["time"] if it["row"]["finished"] else None,
                                                           "reason": it["reason_name"]}) for it in items]}))
    clean = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()}
             for r in rd.load_history()]
    (out / "history.json").write_text(json.dumps(clean, allow_nan=False))
    g, genome = rd.best_genome(gen_model)
    (models / f"{model_name}.json").write_text(json.dumps({
        "name": model_name, "generation": g, "topology": topo.to_dict(),
        "sim": {**sim_cfg.__dict__, "ray_angles_deg": list(sim_cfg.ray_angles_deg)},
        "genome": [float(x) for x in genome]}))
    if generalization_specs:
        rows, summary = benchmark(genome, topo, generalization_specs, sim_cfg, CarParams())
        for r in rows:
            r["reason_name"] = REASON_NAMES[r["reason"]]
        (out / "generalization.json").write_text(json.dumps({"generation": g, "summary": summary, "rows": rows}))
    return out
