"""Registro de pistas: 'interlagos', 'seed:<n>' ou 'csv:<caminho>'."""
from __future__ import annotations

from functools import lru_cache

from ..track import Track


@lru_cache(maxsize=64)
def get_track(spec: str) -> Track:
    spec = spec.strip()
    if spec in ("interlagos", "interlagos_stylized"):
        from .interlagos import build_interlagos
        return build_interlagos()
    if spec.startswith("seed:"):
        from .generator import generate_track
        return generate_track(int(spec.split(":", 1)[1]))
    if spec.startswith("csv:"):
        path = spec.split(":", 1)[1]
        return Track.from_csv(path.rsplit("/", 1)[-1], path)
    raise ValueError(f"pista desconhecida: {spec!r} (use 'interlagos', 'seed:N' ou 'csv:caminho')")


def parse_specs(text: str) -> list[str]:
    """'interlagos,seed:1-3' -> ['interlagos','seed:1','seed:2','seed:3']."""
    out: list[str] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("seed:") and "-" in part[5:]:
            a, b = part[5:].split("-", 1)
            out += [f"seed:{i}" for i in range(int(a), int(b) + 1)]
        else:
            out.append(part)
    return out
