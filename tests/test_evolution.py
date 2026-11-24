import numpy as np

from f1ia.evolution import EvoConfig, Evolution
from f1ia.nn import PopulationPolicy, Topology
from f1ia.sim import Sim
from f1ia.tracks import get_track


def test_topology_counts_and_forward():
    t = Topology(9, (12, 8), 2)
    assert t.n_params == 9 * 12 + 12 + 12 * 8 + 8 + 8 * 2 + 2
    g = t.random_genomes(np.random.default_rng(0), 5)
    out = t.forward(g, np.random.default_rng(1).random((5, 9)))
    assert out.shape == (5, 2) and np.all(np.abs(out) <= 1)


def test_forward_matches_naive():
    t = Topology(3, (4,), 2)
    g = t.random_genomes(np.random.default_rng(0), 1)[0]
    x = np.array([0.3, -0.2, 0.9])
    W1, b1 = g[:12].reshape(3, 4), g[12:16]
    W2, b2 = g[16:24].reshape(4, 2), g[24:26]
    ref = np.tanh(np.tanh(x @ W1 + b1) @ W2 + b2)
    assert np.allclose(t.forward(g[None], x[None])[0], ref)


def test_evolution_keeps_elite_and_size():
    topo = Topology(9, (6,), 2)
    cfg = EvoConfig(pop_size=20, elite=3, immigrants=2)
    evo = Evolution(topo, cfg, seed=1)
    fit = np.arange(20, dtype=float)
    old = evo.population.copy()
    new = evo.next_generation(fit)
    assert new.shape == old.shape
    assert np.allclose(new[:3], old[[19, 18, 17]])            # elite = os 3 melhores, inalterados
    assert not np.allclose(new[3:], old[3:])


def test_evolution_is_reproducible():
    topo = Topology(9, (6,), 2)
    fit = np.random.default_rng(0).random(30)
    a, b = Evolution(topo, EvoConfig(), seed=5), Evolution(topo, EvoConfig(), seed=5)
    assert np.allclose(a.next_generation(fit), b.next_generation(fit))


def test_sigma_decays_and_grows_when_stagnant():
    evo = Evolution(Topology(9, (6,), 2), EvoConfig(), seed=0)
    s0 = evo.sigma
    evo.generation = 40
    assert evo.sigma < s0
    evo.stagnant = 25
    assert evo.sigma > Evolution(Topology(9, (6,), 2), EvoConfig(), seed=0).cfg.mut_sigma_min


def test_evolution_improves_fitness_on_simple_task():
    """Maximizar -||w - alvo||: a evolução tem que chegar perto, independente do simulador."""
    topo = Topology(2, (3,), 1)
    target = np.random.default_rng(3).normal(size=topo.n_params)
    evo = Evolution(topo, EvoConfig(pop_size=80, elite=4, mut_sigma=0.3), seed=0)
    first = None
    for _ in range(150):
        fit = -np.linalg.norm(evo.population - target, axis=1)
        first = fit.max() if first is None else first
        evo.next_generation(fit)
    assert (-np.linalg.norm(evo.population - target, axis=1)).max() > first + 1.0


def test_population_policy_drives_in_sim():
    topo = Topology(9, (12, 8), 2)
    pop = topo.random_genomes(np.random.default_rng(0), 10)
    sim = Sim(get_track("seed:2"), 10).run(PopulationPolicy(pop, topo))
    assert sim.n_alive == 0 and sim.result()["fitness"].shape == (10,)
