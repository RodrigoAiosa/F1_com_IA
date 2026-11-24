import json

import numpy as np
import pytest

from f1ia.compare import compare_generations, format_table, load_run
from f1ia.evaluate import benchmark, run_genome
from f1ia.evolution import EvoConfig
from f1ia.sim import SimConfig
from f1ia.storage import RunDir
from f1ia.train import TrainConfig, Trainer
from f1ia.webexport import export_web


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("runs") / "mini"
    cfg = TrainConfig(run=str(d), tracks=["seed:3"], generations=3, val_tracks=["seed:50"], val_every=2,
                      evo=EvoConfig(pop_size=16, elite=2, immigrants=2), hidden=(8,))
    Trainer(cfg).run()
    return d


def test_checkpoints_and_history(run_dir):
    rd = RunDir(run_dir)
    assert rd.generations() == [1, 2, 3]
    h = rd.load_history()
    assert [r["gen"] for r in h] == [1, 2, 3]
    assert not np.isnan(h[1]["val_progress"])             # validação na geração 2
    ck = rd.load_checkpoint(2)
    assert ck["genomes"].shape[0] == 16 and ck["fitness"][0] >= ck["fitness"][-1]


def test_checkpoint_replays_exactly_like_training(run_dir):
    """O melhor genoma salvo tem que reproduzir o mesmo resultado registrado no treino."""
    rd, topo, cfg = load_run(run_dir)
    for g in rd.generations():
        genome = rd.load_checkpoint(g)["genomes"][0]
        _, r = run_genome(genome, topo, "seed:3", cfg)
        row = next(x for x in rd.load_history() if x["gen"] == g)
        assert r["progress"] == pytest.approx(row["best_progress"])
        assert r["fitness"] == pytest.approx(row["best_fit"], abs=1e-6)


def test_resume_continues_numbering(run_dir, tmp_path):
    import shutil
    d = tmp_path / "copy"
    shutil.copytree(run_dir, d)
    cfg = TrainConfig(run=str(d), tracks=["seed:3"], generations=2, val_tracks=[],
                      evo=EvoConfig(pop_size=16, elite=2, immigrants=2), hidden=(8,))
    t = Trainer(cfg, resume=True)
    assert t.generation == 4
    t.run(2)
    assert RunDir(d).generations() == [1, 2, 3, 4, 5]


def test_refuses_to_overwrite_existing_run(run_dir):
    cfg = TrainConfig(run=str(run_dir), tracks=["seed:3"], val_tracks=[], hidden=(8,))
    with pytest.raises(FileExistsError):
        Trainer(cfg)


def test_workers_match_single_process(tmp_path):
    from f1ia.evaluate import Evaluator
    from f1ia.nn import Topology
    topo = Topology(9, (8,), 2)
    pop = topo.random_genomes(np.random.default_rng(0), 12)
    a = Evaluator(topo, ["seed:2"], workers=1).evaluate(pop)[0]["fitness"]
    ev = Evaluator(topo, ["seed:2"], workers=2)
    b = ev.evaluate(pop)[0]["fitness"]
    ev.close()
    assert np.allclose(a, b)


def test_compare_and_benchmark(run_dir):
    track, items = compare_generations(run_dir, [1, "best"], "seed:9")
    assert len(items) == 2 and items[0]["xy"].shape[1] == 2
    assert "geração" in format_table(items)
    rd, topo, cfg = load_run(run_dir)
    rows, s = benchmark(rd.best_genome()[1], topo, ["seed:60", "seed:61"], cfg)
    assert s["n"] == 2 and 0 <= s["mean_progress"] <= 1


def test_plots(run_dir, tmp_path):
    from f1ia.plots import plot_comparison, plot_history, save_track_png
    from f1ia.tracks import get_track
    rd = RunDir(run_dir)
    assert (tmp_path / "h.png").exists() is False
    plot_history(rd.load_history(), str(tmp_path / "h.png"))
    track, items = compare_generations(run_dir, [1, 3], "seed:3")
    plot_comparison(track, items, str(tmp_path / "c.png"))
    save_track_png(get_track("seed:1"), str(tmp_path / "t.png"))
    assert all((tmp_path / f).stat().st_size > 5000 for f in ("h.png", "c.png", "t.png"))


def test_export_web(run_dir, tmp_path):
    out = export_web(run_dir, tmp_path / "data", "seed:3", [1, "best"], "best", ["seed:70", "seed:71"],
                     tmp_path / "models", "mini")
    rep = json.loads((out / "replays.json").read_text())
    assert len(rep["cars"]) == 2 and rep["track"]["length"] > 1000
    assert json.loads((out / "history.json").read_text())[0]["gen"] == 1
    assert json.loads((out / "generalization.json").read_text())["summary"]["n"] == 2
    m = json.loads((tmp_path / "models" / "mini.json").read_text())
    assert len(m["genome"]) == 9 * 8 + 8 + 8 * 2 + 2
