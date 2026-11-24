"""Algoritmo genético: elitismo, seleção por torneio, cruzamento e mutação gaussiana."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .nn import Topology


@dataclass
class EvoConfig:
    pop_size: int = 60
    elite: int = 4              # melhores copiados sem alteração
    tournament: int = 3
    crossover_rate: float = 0.7
    mut_rate: float = 0.12      # probabilidade de mutar cada gene
    mut_sigma: float = 0.35     # desvio da mutação no início
    mut_sigma_min: float = 0.05
    sigma_decay: float = 0.985  # por geração
    stagnation_gens: int = 10   # sem melhora por tantas gerações -> aumenta a mutação
    immigrants: int = 3         # genomas aleatórios novos por geração (diversidade)
    init_scale: float = 1.0

    def to_dict(self):
        return asdict(self)


class Evolution:
    def __init__(self, topology: Topology, cfg: EvoConfig, seed: int = 0):
        self.topology, self.cfg, self.seed = topology, cfg, seed
        self.generation = 0
        self.best_fitness = -np.inf
        self.stagnant = 0
        self.population = topology.random_genomes(np.random.default_rng([seed, 0]), cfg.pop_size, cfg.init_scale)

    # ------------------------------------------------------------------ parâmetros
    @property
    def sigma(self) -> float:
        c = self.cfg
        s = max(c.mut_sigma_min, c.mut_sigma * c.sigma_decay ** self.generation)
        if self.stagnant >= c.stagnation_gens:
            s = min(c.mut_sigma, s * (1.0 + 0.1 * (self.stagnant - c.stagnation_gens + 1)))
        return float(s)

    # ------------------------------------------------------------------ operadores
    def _select(self, rng, fitness, k):
        cand = rng.integers(0, len(fitness), size=(k, self.cfg.tournament))
        return cand[np.arange(k), fitness[cand].argmax(axis=1)]

    def _crossover(self, rng, a, b):
        """Metade dos genes por escolha uniforme, metade por mistura aritmética (alpha aleatório)."""
        uniform = np.where(rng.random(a.shape) < 0.5, a, b)
        alpha = rng.random((len(a), 1))
        blend = alpha * a + (1 - alpha) * b
        return np.where(rng.random((len(a), 1)) < 0.5, blend, uniform)

    def _mutate(self, rng, g, sigma):
        mask = rng.random(g.shape) < self.cfg.mut_rate
        return g + mask * rng.normal(0.0, sigma, size=g.shape)

    def next_generation(self, fitness: np.ndarray) -> np.ndarray:
        """Recebe o fitness da população atual e devolve a próxima população."""
        c, pop = self.cfg, self.population
        fitness = np.asarray(fitness, dtype=float)
        order = np.argsort(-fitness)
        if fitness[order[0]] > self.best_fitness + 1e-9:
            self.best_fitness, self.stagnant = float(fitness[order[0]]), 0
        else:
            self.stagnant += 1
        rng = np.random.default_rng([self.seed, self.generation + 1])
        sigma = self.sigma
        n_children = c.pop_size - c.elite - c.immigrants
        pa = pop[self._select(rng, fitness, n_children)]
        pb = pop[self._select(rng, fitness, n_children)]
        do_x = rng.random(n_children) < c.crossover_rate
        children = np.where(do_x[:, None], self._crossover(rng, pa, pb), pa)
        children = self._mutate(rng, children, sigma)
        immigrants = self.topology.random_genomes(rng, c.immigrants, c.init_scale)
        self.population = np.concatenate([pop[order[:c.elite]], children, immigrants], axis=0)
        self.generation += 1
        return self.population
