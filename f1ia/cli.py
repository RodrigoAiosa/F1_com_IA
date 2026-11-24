"""Linha de comando: python -m f1ia <comando> ..."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .tracks import get_track, parse_specs


def _tracks_arg(p, default="interlagos"):
    p.add_argument("--tracks", default=default, help="ex.: interlagos | seed:7 | interlagos,seed:1-5 | csv:arquivo.csv")


def cmd_train(a):
    from .evolution import EvoConfig
    from .sim import SimConfig
    from .train import TrainConfig, Trainer, format_row
    evo = EvoConfig(pop_size=a.pop, elite=a.elite, mut_sigma=a.sigma, mut_rate=a.mut_rate)
    cfg = TrainConfig(run=a.run, tracks=parse_specs(a.tracks), seed=a.seed, generations=a.gens,
                      hidden=tuple(int(x) for x in a.hidden.split(",")), workers=a.workers,
                      val_tracks=parse_specs(a.val_tracks) if a.val_tracks else [], val_every=a.val_every,
                      checkpoint_every=a.checkpoint_every, stop_when_finishers=a.stop_at, overwrite=a.overwrite,
                      evo=evo, sim=SimConfig(n_laps=a.laps))
    tr = Trainer(cfg, resume=a.resume)
    print(f"Treinando em {cfg.tracks} | pop {evo.pop_size} | rede {tr.topology.sizes} "
          f"({tr.topology.n_params} pesos) | {cfg.workers} processo(s) | saída: {cfg.run}")
    every = max(1, a.log_every)
    try:
        tr.run(a.gens, callback=lambda r: print(format_row(r), flush=True) if (r["gen"] % every == 0 or r["gen"] == 1) else None)
    except KeyboardInterrupt:
        print("\nInterrompido — checkpoints até a última geração concluída foram salvos.")
    if a.plot:
        from .plots import plot_history
        out = plot_history(tr.rd.load_history(), str(tr.rd.path / "history.png"), f"Treino: {cfg.run}")
        print("Gráfico:", out)


def cmd_play(a):
    from .ui.app import play
    play(a.tracks.split(",")[0], a.mode, a.fps)


def cmd_watch(a):
    from .ui.app import watch
    gens = [g if g in ("best", "last") else int(g) for g in a.gen.split(",")]
    watch(a.run, gens, a.tracks.split(",")[0], a.ghosts)


def cmd_live(a):
    from .ui.app import live
    live(a)


def cmd_compare(a):
    from .compare import compare_generations, format_table
    from .plots import plot_comparison
    gens = [g if g in ("best", "last") else int(g) for g in a.gens.split(",")]
    track, items = compare_generations(a.run, gens, a.tracks.split(",")[0])
    print(f"Pista: {track.name}\n" + format_table(items))
    if a.out:
        print("Gráfico:", plot_comparison(track, items, a.out))


def cmd_evaluate(a):
    from .compare import load_run
    from .evaluate import benchmark
    from .sim import REASON_NAMES
    rd, topo, sim_cfg = load_run(a.run)
    g, genome = rd.best_genome(a.gen if a.gen in ("best", "last") else int(a.gen))
    specs = parse_specs(a.tracks)
    rows, s = benchmark(genome, topo, specs, sim_cfg)
    print(f"Geração {g} em {len(specs)} pista(s):")
    for r in rows:
        t = f"{r['time']:.1f}s" if r["finished"] else "--"
        print(f"  {r['track']:>10}  {r['length'] / 1000:.2f} km  {r['progress'] * 100:5.1f}%  {t:>7}  {REASON_NAMES[r['reason']]}")
    print(f"Completou {s['finish_rate'] * 100:.0f}% das pistas | progresso médio {s['mean_progress'] * 100:.1f}%")
    if a.json:
        json.dump({"generation": g, "rows": rows, "summary": s}, open(a.json, "w"), indent=2)


def cmd_plot(a):
    from .plots import plot_history
    from .storage import RunDir
    rd = RunDir(a.run)
    print("Gráfico:", plot_history(rd.load_history(), a.out or str(rd.path / "history.png"), f"Treino: {a.run}"))


def cmd_track(a):
    from .plots import save_track_png
    for spec in parse_specs(a.tracks):
        t = get_track(spec)
        out = a.out.replace("{}", spec.replace(":", "_")) if a.out else f"{spec.replace(':', '_')}.png"
        save_track_png(t, out)
        print(f"{spec}: {t.length:.0f} m, raio mínimo {1 / np.abs(t.curvature).max():.0f} m -> {out}")


def cmd_export(a):
    from .webexport import export_web
    gens = [g if g in ("best", "last") else int(g) for g in a.gens.split(",")]
    out = export_web(a.run, a.out, a.track, gens, "best", parse_specs(a.generalization), a.model_dir, a.model_name)
    print("Exportado para", out)


def build_parser():
    p = argparse.ArgumentParser(prog="f1ia", description="Neuroevolução de carros de corrida")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="treino em lote, sem janela")
    t.add_argument("--run", default="runs/interlagos", help="pasta de saída (checkpoints, histórico)")
    _tracks_arg(t)
    t.add_argument("--gens", type=int, default=100)
    t.add_argument("--pop", type=int, default=60)
    t.add_argument("--elite", type=int, default=4)
    t.add_argument("--sigma", type=float, default=0.35, help="desvio inicial da mutação")
    t.add_argument("--mut-rate", type=float, default=0.12)
    t.add_argument("--hidden", default="12,8")
    t.add_argument("--laps", type=int, default=1)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--workers", type=int, default=1, help="processos em paralelo")
    t.add_argument("--val-tracks", default="seed:1000-1005", help="pistas de validação (generalização); '' desliga")
    t.add_argument("--val-every", type=int, default=5)
    t.add_argument("--checkpoint-every", type=int, default=1)
    t.add_argument("--stop-at", type=float, default=None, help="para quando esta fração da população completar (0-1)")
    t.add_argument("--log-every", type=int, default=1)
    t.add_argument("--resume", action="store_true", help="continua de onde parou")
    t.add_argument("--overwrite", action="store_true", help="apaga um treino existente nesta pasta")
    t.add_argument("--plot", action="store_true", help="gera history.png ao final")
    t.set_defaults(fn=cmd_train)

    pl = sub.add_parser("play", help="dirigir manualmente (setas/WASD) ou ver o piloto automático")
    _tracks_arg(pl)
    pl.add_argument("--mode", choices=["manual", "auto"], default="manual")
    pl.add_argument("--fps", type=int, default=60)
    pl.set_defaults(fn=cmd_play)

    w = sub.add_parser("watch", help="replay de uma geração (ou várias, como fantasmas)")
    w.add_argument("--run", required=True)
    w.add_argument("--gen", default="best", help="N | best | last | lista: 1,10,best")
    _tracks_arg(w)
    w.add_argument("--ghosts", action="store_true", help="mostra todas as gerações listadas ao mesmo tempo")
    w.set_defaults(fn=cmd_watch)

    lv = sub.add_parser("live", help="treino com visualização ao vivo")
    lv.add_argument("--run", default="runs/live")
    _tracks_arg(lv)
    lv.add_argument("--pop", type=int, default=40)
    lv.add_argument("--seed", type=int, default=0)
    lv.add_argument("--hidden", default="12,8")
    lv.set_defaults(fn=cmd_live)

    c = sub.add_parser("compare", help="compara gerações na mesma pista")
    c.add_argument("--run", required=True)
    c.add_argument("--gens", default="1,5,10,best")
    _tracks_arg(c)
    c.add_argument("--out", default=None, help="PNG com trajetórias sobrepostas")
    c.set_defaults(fn=cmd_compare)

    e = sub.add_parser("evaluate", help="testa generalização em pistas novas")
    e.add_argument("--run", required=True)
    e.add_argument("--gen", default="best")
    _tracks_arg(e, "seed:100-119")
    e.add_argument("--json", default=None)
    e.set_defaults(fn=cmd_evaluate)

    g = sub.add_parser("plot", help="gráficos de evolução do treino")
    g.add_argument("--run", required=True)
    g.add_argument("--out", default=None)
    g.set_defaults(fn=cmd_plot)

    tk = sub.add_parser("track", help="gera PNG de pistas (inclusive por semente)")
    _tracks_arg(tk, "seed:1-4")
    tk.add_argument("--out", default=None, help="use {} para o nome da pista")
    tk.set_defaults(fn=cmd_track)

    x = sub.add_parser("export-web", help="exporta dados para o visualizador web (Vercel)")
    x.add_argument("--run", required=True)
    x.add_argument("--out", default="public/data")
    x.add_argument("--track", default="interlagos")
    x.add_argument("--gens", default="1,5,10,best")
    x.add_argument("--generalization", default="seed:100-119")
    x.add_argument("--model-dir", default="models")
    x.add_argument("--model-name", default="interlagos_best")
    x.set_defaults(fn=cmd_export)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
