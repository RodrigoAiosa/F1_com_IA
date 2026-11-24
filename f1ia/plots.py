"""Gráficos com matplotlib (backend Agg: funciona sem janela)."""
from __future__ import annotations

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_track(ax, track, trajectories=(), show_start=True):
    """trajectories: iterável de (xy (T,2), rótulo, cor)."""
    closed = lambda a: np.vstack([a, a[:1]])
    ax.fill(*np.vstack([closed(track.left), closed(track.right)[::-1]]).T, color="#d9d9d9", lw=0)
    ax.plot(*closed(track.left).T, color="#333", lw=0.8)
    ax.plot(*closed(track.right).T, color="#333", lw=0.8)
    for xy, label, color in trajectories:
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.6, label=label)
    if show_start:
        p, t = track.start_pos, track.tangent[0]
        ax.plot(*p, "o", color="green", ms=5)
        ax.annotate("", xy=p + t * 60, xytext=p, arrowprops=dict(arrowstyle="->", color="green"))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{track.name} — {track.length / 1000:.2f} km")


def plot_history(history: list[dict], out: str, title: str = "Evolução do treino"):
    plt = _plt()
    g = [r["gen"] for r in history]
    col = lambda k: np.array([r.get(k, np.nan) for r in history], dtype=float)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    fig.suptitle(title)
    a = ax[0, 0]
    a.plot(g, col("best_fit"), label="melhor", lw=2)
    a.plot(g, col("mean_fit"), label="média")
    a.plot(g, col("median_fit"), label="mediana", alpha=0.7)
    a.fill_between(g, col("mean_fit") - col("std_fit"), col("mean_fit") + col("std_fit"), alpha=0.15)
    a.set(title="Fitness (1.0 = volta completa)", xlabel="geração")
    a.legend()
    a = ax[0, 1]
    a.plot(g, col("best_progress") * 100, label="melhor (%)", lw=2)
    a.plot(g, col("mean_progress") * 100, label="média (%)")
    a.set(title="Progresso na pista de treino", xlabel="geração", ylabel="% da volta")
    a2 = a.twinx()
    a2.plot(g, col("n_finishers"), color="tab:green", label="completaram", alpha=0.8)
    a2.set_ylabel("carros que completaram")
    a.legend(loc="lower right")
    a = ax[1, 0]
    t = col("best_time")
    a.plot(g, t, ".-", color="tab:purple")
    a.set(title="Melhor tempo de volta (s)", xlabel="geração")
    a = ax[1, 1]
    v, f = col("val_progress"), col("val_finish_rate")
    ok = ~np.isnan(v)
    if ok.any():
        a.plot(np.array(g)[ok], v[ok] * 100, "o-", label="progresso médio (%)")
        a.plot(np.array(g)[ok], f[ok] * 100, "s--", label="pistas completas (%)")
        a.set_ylim(0, 105)
        a.legend()
    a.set(title="Generalização: pistas novas (nunca vistas no treino)", xlabel="geração")
    a3 = a.twinx()
    a3.plot(g, col("sigma"), color="gray", alpha=0.5)
    a3.set_ylabel("σ da mutação", color="gray")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_comparison(track, items, out: str):
    """items: lista de dicts {label, xy, row}. Trajetórias sobrepostas + barras de progresso."""
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    n = max(len(items) - 1, 1)
    trajs = [(it["xy"], it["label"], cmap(i / n)) for i, it in enumerate(items)]
    plot_track(ax[0], track, trajs)
    ax[0].legend(loc="upper left", fontsize=8)
    labels = [it["label"] for it in items]
    prog = [it["row"]["progress"] * 100 for it in items]
    colors = [cmap(i / n) for i in range(len(items))]
    ax[1].barh(labels, prog, color=colors)
    for i, it in enumerate(items):
        r = it["row"]
        txt = f"{r['time']:.1f}s" if r["finished"] else it["reason_name"]
        ax[1].text(min(prog[i] + 1, 70), i, txt, va="center", fontsize=8)
    ax[1].set(xlim=(0, 110), xlabel="% da volta", title="Progresso")
    ax[1].invert_yaxis()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def save_track_png(track, out: str):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_track(ax, track)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out
