"""Paths and secrets. Secrets come from the environment or a local .env that is never committed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("STOCK_AGENT_DATA", ROOT / "data"))
RAW = DATA / "raw"
CLEAN = DATA / "clean"


def load_env(path: Path = ROOT / ".env") -> None:
    """Load KEY='value' lines into os.environ without overriding values already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        os.environ.setdefault(k.strip(), v)


load_env()
