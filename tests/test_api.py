import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from f1ia.nn import Topology
from f1ia.sim import SimConfig


@pytest.fixture()
def client(tmp_path, monkeypatch):
    topo = Topology(9, (8,), 2)
    genome = topo.random_genomes(np.random.default_rng(0), 1)[0]
    cfg = SimConfig()
    (tmp_path / "demo.json").write_text(json.dumps({
        "name": "demo", "generation": 3, "topology": topo.to_dict(),
        "sim": {**cfg.__dict__, "ray_angles_deg": list(cfg.ray_angles_deg)}, "genome": genome.tolist()}))
    monkeypatch.setattr(app_module, "MODELS_DIR", tmp_path)
    app_module._load_model.cache_clear()
    return TestClient(app_module.app)


def test_health_and_models(client):
    assert client.get("/api/health").json() == {"ok": True}
    assert client.get("/api/models").json()["models"][0]["name"] == "demo"


def test_track_endpoints(client):
    d = client.get("/api/track?track=interlagos").json()
    assert 4100 < d["length"] < 4500 and len(d["center"]) == len(d["left"]) == len(d["right"])
    assert client.get("/api/track?track=seed:12").json()["name"] == "seed:12"


@pytest.mark.parametrize("bad", ["csv:/etc/passwd", "seed:abc", "../x", "seed:99999999999"])
def test_rejects_unsafe_track_specs(client, bad):
    assert client.get("/api/track", params={"track": bad}).status_code == 400


def test_simulate_autopilot_and_model(client):
    d = client.get("/api/simulate?track=seed:3&model=autopilot").json()
    assert d["result"]["finished"] and len(d["car"]["frames"]) > 300
    m = client.get("/api/simulate?track=seed:3&model=demo").json()
    assert 0 <= m["result"]["progress"] <= 1 and m["car"]["label"].startswith("demo")


def test_model_not_found_and_traversal(client):
    assert client.get("/api/simulate?model=nao_existe").status_code == 404
    assert client.get("/api/simulate", params={"model": "../app"}).status_code == 404


def test_api_matches_local_replay(client):
    from f1ia.evaluate import run_genome
    d, topo, cfg, genome = app_module._load_model("demo")
    _, r = run_genome(genome[0], topo, "seed:3", cfg)
    api = client.get("/api/simulate?track=seed:3&model=demo").json()["result"]
    assert api["progress"] == pytest.approx(float(r["progress"]))
