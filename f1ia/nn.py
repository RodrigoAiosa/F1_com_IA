"""Rede neural pequena (MLP, tanh) vetorizada para uma população inteira.

O genoma é um vetor plano com [W1, b1, W2, b2, ...]. Isso torna cruzamento e mutação triviais
e permite salvar a população inteira em uma única matriz (P, n_params).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class Topology:
    n_in: int = 9
    hidden: tuple = (12, 8)
    n_out: int = 2

    @property
    def sizes(self):
        return (self.n_in, *self.hidden, self.n_out)

    @property
    def n_params(self) -> int:
        s = self.sizes
        return sum(a * b + b for a, b in zip(s[:-1], s[1:]))

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(int(d["n_in"]), tuple(int(h) for h in d["hidden"]), int(d["n_out"]))

    def random_genomes(self, rng: np.random.Generator, n: int, scale: float = 1.0) -> np.ndarray:
        """Inicialização estilo Xavier por camada."""
        parts = []
        s = self.sizes
        for a, b in zip(s[:-1], s[1:]):
            parts.append(rng.normal(0, scale / np.sqrt(a), size=(n, a * b)))
            parts.append(np.zeros((n, b)))
        return np.concatenate(parts, axis=1)

    def forward(self, genomes: np.ndarray, x: np.ndarray) -> np.ndarray:
        """genomes (k, n_params), x (k, n_in) -> (k, n_out) em [-1, 1]."""
        k = len(genomes)
        s = self.sizes
        off = 0
        h = x
        for a, b in zip(s[:-1], s[1:]):
            W = genomes[:, off:off + a * b].reshape(k, a, b)
            off += a * b
            bias = genomes[:, off:off + b]
            off += b
            h = np.tanh(np.matmul(h[:, None, :], W)[:, 0, :] + bias)
        return h


class PopulationPolicy:
    """Controla cada carro da simulação com o seu próprio genoma."""

    def __init__(self, genomes: np.ndarray, topology: Topology):
        self.genomes = np.asarray(genomes, dtype=np.float64)
        self.topology = topology

    def act(self, sim, alive_idx):
        out = self.topology.forward(self.genomes[alive_idx], sim.observe(alive_idx))
        return out[:, 0], out[:, 1]
