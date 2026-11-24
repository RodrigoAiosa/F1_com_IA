"""Comparação de gerações: roda o melhor carro de cada geração na MESMA pista."""
from __future__ import annotations

from .evaluate import run_genome
from .nn import Topology
from .sim import CarParams, REASON_NAMES, SimConfig
from .storage import RunDir
from .tracks import get_track


def load_run(path):
    rd = RunDir(path)
    cfg = rd.load_config()
    topo = Topology(cfg["sim"]["ray_angles_deg"].__len__() + 2, tuple(cfg["hidden"]), 2)
    sim_cfg = SimConfig(**{**cfg["sim"], "ray_angles_deg": tuple(cfg["sim"]["ray_angles_deg"])})
    return rd, topo, sim_cfg


def compare_generations(run_path, gens, spec: str, record_stride: int = 2):
    """Retorna lista de dicts {gen, label, xy (T,2), row, reason_name, traj (T,4)}."""
    rd, topo, sim_cfg = load_run(run_path)
    items = []
    for w in gens:
        g = rd.resolve_gen(w)
        genome = rd.load_checkpoint(g)["genomes"][0]
        sim, r = run_genome(genome, topo, spec, sim_cfg, CarParams(), record_stride=record_stride)
        traj = sim.trajectory()[:, 0, :]
        items.append({"gen": g, "label": f"ger {g}", "xy": traj[:, :2], "traj": traj,
                      "row": {"progress": float(r["progress"]), "finished": bool(r["finished"]),
                              "time": float(r["time"]), "avg_speed": float(r["avg_speed"])},
                      "reason_name": REASON_NAMES[int(r["reason"])]})
    return get_track(spec), items


def format_table(items) -> str:
    lines = [f"{'geração':>8} {'progresso':>10} {'volta':>8} {'vel.média':>10}  resultado"]
    for it in items:
        r = it["row"]
        t = f"{r['time']:.1f}s" if r["finished"] else "--"
        lines.append(f"{it['gen']:>8} {r['progress'] * 100:>9.1f}% {t:>8} {r['avg_speed'] * 3.6:>7.0f} km/h  {it['reason_name']}")
    return "\n".join(lines)
