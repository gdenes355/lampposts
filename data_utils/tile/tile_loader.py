import os
import re

from data_utils.models import Tile

_SQUARE_ORIGINS = {
    "tl": (500_000, 200_000),
}

_cache: dict[str, Tile] = {}


def load_tile(path: str) -> Tile:
    """Load a Tile from a file path, returning a cached instance if already loaded."""
    key = os.path.abspath(path)
    if key in _cache:
        return _cache[key]
    m = re.search(r"(tl)(\d{3})(\d{3})", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse grid ref from: {path}")
    origin_e, origin_n = _SQUARE_ORIGINS[m.group(1)]
    left = origin_e + int(m.group(2)) * 100
    bottom = origin_n + int(m.group(3)) * 100
    tile = Tile(path=key, bbox_bng=(left, bottom, left + 100, bottom + 100))
    _cache[key] = tile
    return tile


def find_tile(x: float, y: float, data_dir: str = "data") -> Tile | None:
    """Return the Tile containing BNG coordinate (x, y), or None."""
    for square, (origin_e, origin_n) in _SQUARE_ORIGINS.items():
        e_idx = int((x - origin_e) // 100)
        n_idx = int((y - origin_n) // 100)
        pattern = f"{square}{e_idx:03d}{n_idx:03d}"
        for subdir in ("lampposts", "no_lampposts"):
            folder = os.path.join(data_dir, subdir)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if pattern in fname and fname.endswith(".tif"):
                    return load_tile(os.path.join(folder, fname))
    return None


if __name__ == "__main__":
    tile = load_tile("data/lampposts/camb-tl439579-1.tif")
    print(tile)