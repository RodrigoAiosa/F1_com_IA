import numpy as np
import pytest

from f1ia.track import Track, wrap_angle
from f1ia.tracks import get_track, parse_specs
from f1ia.tracks.generator import generate_track


def circle_track(r=300.0, width=14.0):
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    return Track("circulo", np.stack([r * np.cos(t), r * np.sin(t)], axis=1), width)


def test_interlagos_length_and_curvature():
    t = get_track("interlagos")
    assert 4100 < t.length < 4500
    assert 1 / np.abs(t.curvature).max() > 20      # sem curvas impossíveis
    assert t.M > 1000


def test_circle_geometry():
    t = circle_track()
    assert t.length == pytest.approx(2 * np.pi * 300, rel=1e-3)
    assert np.allclose(np.abs(t.curvature), 1 / 300, rtol=0.03)
    assert t.curvature.mean() > 0                    # anti-horário => curva à esquerda positiva


def test_on_track_and_rays():
    t = circle_track()
    c = t.center[10]
    assert t.on_track(np.array([c[0]]), np.array([c[1]]))[0]
    assert not t.on_track(np.array([0.0]), np.array([0.0]))[0]          # centro do círculo
    assert not t.on_track(np.array([1e6]), np.array([1e6]))[0]          # fora da grade
    # raio apontando para fora a partir da linha central: borda a ~ (largura/2 - raio do carro)
    n = t.normal[10]
    ang = np.array([[np.arctan2(-n[1], -n[0])]])                        # para fora (direita)
    d = t.raycast(np.array([c[0]]), np.array([c[1]]), ang, 100, 0.5)[0, 0]
    assert d == pytest.approx(14 / 2 - t.car_radius, abs=1.5)


def test_locate_tracks_progress():
    t = circle_track()
    idx = np.array([0])
    p = t.center[5] + t.normal[5] * 3.0
    i, s, lat = t.locate(np.array([p[0]]), np.array([p[1]]), idx)
    assert i[0] == 5 and lat[0] == pytest.approx(3.0, abs=0.3)
    assert s[0] == pytest.approx(t.s[5], abs=1.0)


def test_wrap_angle():
    assert wrap_angle(3 * np.pi / 2) == pytest.approx(-np.pi / 2)


def test_seed_is_deterministic_and_varied():
    a, b, c = generate_track(3), generate_track(3), generate_track(4)
    assert np.allclose(a.center, b.center)
    assert a.length != pytest.approx(c.length, rel=1e-3)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42, 1000])
def test_generated_tracks_are_valid(seed):
    t = generate_track(seed)
    assert 2500 < t.length < 6000
    assert 1 / np.abs(t.curvature).max() >= 29            # raio mínimo respeitado
    assert t.grid.sum() > 1000


def test_specs():
    assert parse_specs("interlagos,seed:1-3") == ["interlagos", "seed:1", "seed:2", "seed:3"]
    with pytest.raises(ValueError):
        get_track("marte")
