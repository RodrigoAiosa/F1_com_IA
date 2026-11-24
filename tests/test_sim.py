import numpy as np
import pytest

from f1ia.autopilot import Autopilot
from f1ia.sim import FINISHED, OFF_TRACK, STALLED, CarParams, Sim, SimConfig
from f1ia.tracks import get_track


class Const:
    def __init__(self, steer, thr):
        self.s, self.t = steer, thr

    def act(self, sim, ai):
        return np.full(len(ai), self.s), np.full(len(ai), self.t)


def test_idle_car_stalls():
    sim = Sim(get_track("interlagos"), 1).run(Const(0.0, 0.0))
    assert sim.reason[0] == STALLED and sim.result()["progress"][0] < 0.01


def test_full_lock_leaves_track():
    sim = Sim(get_track("interlagos"), 3).run(Const(1.0, 1.0))
    assert (sim.reason == OFF_TRACK).all()


def test_speed_is_bounded_and_accelerates():
    sim = Sim(get_track("seed:0"), 1)
    for _ in range(40):
        sim.step(sim.alive_idx, np.zeros(1), np.ones(1))
    assert 0 < sim.v[0] <= sim.car.max_speed
    v1 = sim.v[0]
    for _ in range(20):
        sim.step(sim.alive_idx, np.zeros(1), -np.ones(1))
    assert sim.v[0] < v1


def test_lateral_grip_limits_turning():
    """Em alta velocidade o raio mínimo de curva é v²/a_lat, não o do volante."""
    tr = get_track("seed:0")
    sim = Sim(tr, 1, cfg=SimConfig(stall_time=1e9))
    sim.v[:] = 40.0
    h0 = sim.h[0]
    sim.steer[:] = 1.0
    sim.step(sim.alive_idx, np.ones(1), np.ones(1))
    omega = abs(sim.h[0] - h0) / sim.cfg.dt
    free = sim.v[0] * sim.car.kmax                      # taxa de giro sem limite de aderência
    assert omega == pytest.approx(sim.car.a_lat / sim.v[0], rel=1e-6) and omega < free / 5


def test_simulation_is_deterministic():
    tr = get_track("seed:5")
    r1 = Sim(tr, 1).run(Autopilot()).result()
    r2 = Sim(tr, 1).run(Autopilot()).result()
    assert r1["time"][0] == r2["time"][0] and r1["fitness"][0] == r2["fitness"][0]


@pytest.mark.parametrize("spec", ["interlagos", "seed:1", "seed:3", "seed:9"])
def test_autopilot_completes_lap(spec):
    sim = Sim(get_track(spec), 1).run(Autopilot())
    r = sim.result()
    assert sim.reason[0] == FINISHED and r["progress"][0] == 1.0
    assert 60 < r["time"][0] < 200


def test_recording_and_independent_check_stays_on_asphalt():
    tr = get_track("interlagos")
    sim = Sim(tr, 1, record_stride=2).run(Autopilot())
    xy = sim.trajectory()[:, 0, :2]
    d = np.linalg.norm(xy[:, None, :] - tr.center[None, ::3, :], axis=2).min(axis=1)
    assert d.max() < tr.width.max() / 2          # nunca saiu da pista (checagem por força bruta)
    assert len(xy) > 500


def test_wrong_way_and_progress_accumulates_across_start_line():
    tr = get_track("interlagos")
    sim = Sim(tr, 1, cfg=SimConfig(n_laps=2)).run(Autopilot())
    assert sim.result()["progress"][0] == 1.0 and sim.best[0] >= 2 * tr.length
