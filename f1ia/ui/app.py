"""Modos de janela: dirigir (manual/auto), replay de gerações e treino ao vivo."""
from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from ..autopilot import Autopilot
from ..sim import CarParams, REASON_NAMES, Sim, SimConfig
from ..tracks import get_track
from .render import Camera, Hud, TrackSurface, draw_car, draw_rays

SIZE = (1200, 760)
PALETTE = [(255, 80, 80), (255, 170, 40), (240, 230, 60), (90, 220, 110), (70, 200, 230), (110, 130, 255), (200, 110, 255)]


def _init(headless: bool = False, size=SIZE, title="F1 com IA"):
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(title)
    return screen, pygame.time.Clock(), Hud()


def _common_keys(event, cam):
    """Teclas comuns. Retorna True se o evento foi tratado."""
    if event.type != pygame.KEYDOWN:
        return False
    if event.key == pygame.K_TAB:
        cam.toggle()
    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
        cam.zoom_by(1.25)
    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        cam.zoom_by(0.8)
    else:
        return False
    return True


def _fmt_time(t):
    return "--" if t is None or np.isnan(t) else f"{int(t // 60)}:{t % 60:05.2f}"


# ---------------------------------------------------------------------------------------
# Dirigir: manual ou piloto automático
# ---------------------------------------------------------------------------------------
def play(spec="interlagos", mode="manual", fps=60, max_frames=None, headless=False):
    screen, clock, hud = _init(headless, title=f"F1 com IA — {mode}")
    track = get_track(spec)
    ts = TrackSurface(track)
    cam = Camera(ts, SIZE)
    cfg = SimConfig(n_laps=3, stall_time=1e9)
    car = CarParams()
    pilot = Autopilot()
    sim = Sim(track, 1, car, cfg)
    show_rays, acc, frames, best_lap = True, 0.0, 0, None
    lap_marks: list[float] = []
    running = True
    while running:
        dt_real = clock.tick(fps) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    sim.reset(); lap_marks.clear(); acc = 0.0
                elif e.key == pygame.K_m:
                    mode = "auto" if mode == "manual" else "manual"
                elif e.key == pygame.K_h:
                    show_rays = not show_rays
                else:
                    _common_keys(e, cam)
        keys = pygame.key.get_pressed()
        acc += min(dt_real, 0.1)
        while acc >= cfg.dt:
            acc -= cfg.dt
            ai = sim.alive_idx
            if len(ai):
                if mode == "auto":
                    steer, thr = pilot.act(sim, ai)
                else:
                    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
                    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
                    up = keys[pygame.K_UP] or keys[pygame.K_w]
                    down = keys[pygame.K_DOWN] or keys[pygame.K_s] or keys[pygame.K_SPACE]
                    steer = np.array([float(left) - float(right)])
                    thr = np.array([1.0 if up else (-1.0 if down else 0.0)])
                before = int(sim.best[0] // track.length)
                sim.step(ai, steer, thr)
                after = int(sim.best[0] // track.length)
                if after > before:
                    lap_marks.append(sim.time)
                    lt = lap_marks[-1] - (lap_marks[-2] if len(lap_marks) > 1 else 0.0)
                    best_lap = lt if best_lap is None else min(best_lap, lt)
        cam.look_at(sim.x[0], sim.y[0])
        cam.draw_track(screen)
        if show_rays and sim.alive[0]:
            draw_rays(screen, cam, sim)
        draw_car(screen, cam, sim.x[0], sim.y[0], sim.h[0], (230, 40, 40), car=car)
        laps = min(sim.best[0] / track.length, cfg.n_laps)
        hud.lines(screen, [f"{track.name}   modo: {mode.upper()}", f"velocidade {sim.v[0] * 3.6:5.0f} km/h",
                           f"volta {min(int(laps) + 1, cfg.n_laps)}/{cfg.n_laps}   progresso {laps / cfg.n_laps * 100:5.1f}%",
                           f"tempo {_fmt_time(sim.time)}   melhor volta {_fmt_time(best_lap)}"])
        hud.text(screen, "setas/WASD dirigem · M troca manual/auto · R reinicia · TAB câmera · +/- zoom · H sensores · ESC sai",
                 (12, SIZE[1] - 26), font=hud.small)
        if not sim.alive[0]:
            r = int(sim.reason[0])
            hud.center_msg(screen, "Volta completa!" if sim.reason[0] == 4 else REASON_NAMES[r].capitalize(),
                           "pressione R para reiniciar")
        pygame.display.flip()
        frames += 1
        if max_frames and frames >= max_frames:
            break
    pygame.quit()
    return sim


# ---------------------------------------------------------------------------------------
# Replay de gerações (carro único ou "fantasmas" de várias gerações)
# ---------------------------------------------------------------------------------------
def watch(run, gens, spec="interlagos", ghosts=False, max_frames=None, headless=False):
    from ..compare import load_run
    from ..evaluate import run_genome

    rd, topo, sim_cfg = load_run(run)
    screen, clock, hud = _init(headless, title=f"Replay — {run}")
    track = get_track(spec)
    ts = TrackSurface(track)
    cam = Camera(ts, SIZE)
    car = CarParams()
    cache: dict[int, dict] = {}
    all_gens = rd.generations()

    def load(g):
        if g not in cache:
            genome = rd.load_checkpoint(g)["genomes"][0]
            sim, r = run_genome(genome, topo, spec, sim_cfg, car, record_stride=1)
            cache[g] = {"gen": g, "traj": sim.trajectory()[:, 0, :], "res": r}
        return cache[g]

    listed = [rd.resolve_gen(g) for g in gens]
    current = listed[0]
    speed, t, paused, frames = 1.0, 0.0, False, 0
    running = True
    while running:
        dt_real = clock.tick(60) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    t = 0.0
                elif e.key == pygame.K_g:
                    ghosts = not ghosts
                elif e.key == pygame.K_UP:
                    speed = min(speed * 2, 16)
                elif e.key == pygame.K_DOWN:
                    speed = max(speed / 2, 0.25)
                elif e.key == pygame.K_RIGHT:
                    t += 5
                elif e.key == pygame.K_LEFT:
                    t = max(0.0, t - 5)
                elif e.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                    d = -1 if e.key == pygame.K_LEFTBRACKET else 1
                    i = max(0, min(len(all_gens) - 1, all_gens.index(current) + d))
                    current, t = all_gens[i], 0.0
                else:
                    _common_keys(e, cam)
        if not paused:
            t += dt_real * speed
        show = [load(g) for g in (listed if ghosts else [current])]
        fi = lambda it: min(int(t / sim_cfg.dt), len(it["traj"]) - 1)
        lead = show[0]
        cam.look_at(*lead["traj"][fi(lead)][:2])
        cam.draw_track(screen)
        for k, it in reversed(list(enumerate(show))):
            x, y, h, _ = it["traj"][fi(it)]
            col = PALETTE[k % len(PALETTE)] if len(show) > 1 else (230, 40, 40)
            draw_car(screen, cam, x, y, h, col, car=car)
            if len(show) > 1:
                sx, sy = cam.to_screen(x, y)
                hud.text(screen, f"ger {it['gen']}", (sx + 10, sy - 18), col, font=hud.small)
        r = lead["res"]
        status = "completou em " + _fmt_time(r["time"]) if r["finished"] else f"{REASON_NAMES[int(r['reason'])]} a {r['progress'] * 100:.0f}%"
        hud.lines(screen, [f"{run}   {track.name}", f"geração {lead['gen']} ({status})",
                           f"tempo {_fmt_time(t)}   velocidade {speed:g}x" + ("   [PAUSA]" if paused else ""),
                           "modo fantasmas" if ghosts else "um carro"])
        hud.text(screen, "ESPAÇO pausa · ↑/↓ velocidade · ←/→ ±5s · [ ] geração anterior/próxima · G fantasmas · R reinicia · TAB câmera",
                 (12, SIZE[1] - 26), font=hud.small)
        pygame.display.flip()
        frames += 1
        if max_frames and frames >= max_frames:
            break
    pygame.quit()


# ---------------------------------------------------------------------------------------
# Treino ao vivo
# ---------------------------------------------------------------------------------------
def live(args, max_frames=None, headless=False, max_gens=None):
    from ..evolution import EvoConfig
    from ..nn import PopulationPolicy
    from ..tracks import parse_specs
    from ..train import TrainConfig, Trainer

    specs = parse_specs(args.tracks)
    cfg = TrainConfig(run=args.run, tracks=specs, seed=args.seed, hidden=tuple(int(x) for x in args.hidden.split(",")),
                      val_tracks=[], evo=EvoConfig(pop_size=args.pop))
    trainer = Trainer(cfg, resume=True)
    screen, clock, hud = _init(headless, title="Treino ao vivo")
    car = CarParams()
    ts_cache: dict[str, tuple] = {}

    def view(spec):
        if spec not in ts_cache:
            tr = get_track(spec)
            ts = TrackSurface(tr)
            ts_cache[spec] = (tr, ts, Camera(ts, SIZE))
        return ts_cache[spec]

    def new_sim():
        spec = specs[(trainer.generation - 1) % len(specs)]   # várias pistas: uma por geração
        tr, ts, cam = view(spec)
        return spec, Sim(tr, len(trainer.evo.population), car, cfg.sim), PopulationPolicy(trainer.evo.population, trainer.topology), cam

    spec, sim, policy, cam = new_sim()
    steps_per_frame, paused, frames, gens_done = 2, False, 0, 0
    hist = trainer.rd.load_history()
    cam.fit = True
    running = True
    while running:
        clock.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif pygame.K_1 <= e.key <= pygame.K_5:
                    steps_per_frame = (1, 2, 6, 20, 120)[e.key - pygame.K_1]
                else:
                    _common_keys(e, cam)
        if not paused:
            for _ in range(steps_per_frame):
                ai = sim.alive_idx
                if len(ai) == 0:
                    break
                steer, thr = policy.act(sim, ai)
                sim.step(ai, steer, thr)
            if sim.n_alive == 0:
                row = trainer.complete_generation([sim.result()])
                hist.append(row)
                gens_done += 1
                if max_gens and gens_done >= max_gens:
                    break
                spec, sim, policy, cam = new_sim()
                cam.fit = cam.fit
        tr = sim.track
        al = sim.alive
        lead = int(np.argmax(np.where(al, sim.best, -1e9))) if al.any() else int(np.argmax(sim.best))
        cam.look_at(sim.x[lead], sim.y[lead])
        cam.draw_track(screen)
        order = np.argsort(sim.best)
        for i in order:
            if not al[i]:
                draw_car(screen, cam, sim.x[i], sim.y[i], sim.h[i], (120, 120, 120), alpha=70, car=car)
        for rank, i in enumerate(order[::-1]):
            if al[i]:
                col = (255, 215, 0) if rank == 0 else (60, 160, 255)
                draw_car(screen, cam, sim.x[i], sim.y[i], sim.h[i], col, car=car)
        if al[lead]:
            draw_rays(screen, cam, sim, lead)
        last = hist[-1] if hist else None
        hud.lines(screen, [f"Geração {trainer.generation}   pista {tr.name}", f"vivos {sim.n_alive}/{sim.n}   tempo {sim.time:5.1f}s",
                           f"líder {sim.best[lead] / tr.length * 100:5.1f}% da volta",
                           f"velocidade da simulação: {steps_per_frame} passo(s)/quadro" + ("   [PAUSA]" if paused else "")])
        if last:
            hud.lines(screen, [f"geração anterior: melhor {last['best_fit']:.2f} · {int(last['n_finishers'])} completaram"], y=100)
        series = [([r["best_fit"] for r in hist], (255, 215, 0)), ([r["mean_fit"] for r in hist], (60, 160, 255))]
        hud.chart(screen, pygame.Rect(SIZE[0] - 310, 10, 300, 130), series, "fitness: melhor (amarelo) / média (azul)")
        hud.text(screen, "ESPAÇO pausa · 1-5 velocidade · TAB câmera (ajuste/seguir líder) · +/- zoom · ESC sai (checkpoints já salvos)",
                 (12, SIZE[1] - 26), font=hud.small)
        pygame.display.flip()
        frames += 1
        if max_frames and frames >= max_frames:
            break
    pygame.quit()
    return trainer
