import types

import pytest

pytest.importorskip("pygame")


def test_play_auto_runs_headless():
    from f1ia.ui.app import play
    sim = play("seed:3", "auto", max_frames=30, headless=True)
    assert sim.step_count > 0


def test_watch_and_live_headless(tmp_path):
    from f1ia.ui import app
    ns = types.SimpleNamespace(run=str(tmp_path / "live"), tracks="seed:3", pop=8, seed=0, hidden="8")
    tr = app.live(ns, max_frames=3000, headless=True, max_gens=2)
    assert tr.generation == 3
    app.watch(str(tmp_path / "live"), [1, "last"], "seed:3", ghosts=True, max_frames=20, headless=True)


def test_cli_parser():
    from f1ia.cli import build_parser
    a = build_parser().parse_args(["train", "--gens", "3", "--tracks", "interlagos,seed:1-2"])
    assert a.gens == 3 and a.tracks.endswith("seed:1-2")
