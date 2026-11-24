"""Ponto de entrada da função Python na Vercel: reaproveita a app FastAPI da raiz."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402,F401
