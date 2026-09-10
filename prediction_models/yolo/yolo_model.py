import shutil
import tempfile
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from data_utils.models import Point, System, Tile

# YOLO11 nano — best speed/accuracy tradeoff for a small dataset
_BASE_WEIGHTS = "yolo11n.pt"

# ~1 m at 3 cm/px (3000 px = 100 m)
_DEFAULT_BBOX_PX = 30


class YoloModel:
    def __init__(self, weights: str = _BASE_WEIGHTS):
        self._weights = weights
        self._model = YOLO(weights)

    def train(
        self,
        train_tiles: list[Tile],
        points: list[Point],
        val_tiles: list[Tile] | None = None,
        epochs: int = 50,
        imgsz: int = 640,
        bbox_px: int = _DEFAULT_BBOX_PX,
        weights_dir: str = "weights/yolo",
        name: str = "train",
    ) -> Path:
        """Train on train_tiles; val_tiles defaults to 20 % of train_tiles if omitted.

        Returns the path to the saved best.pt weights file.
        """
        if val_tiles is None:
            split = max(1, len(train_tiles) // 5)
            val_tiles = train_tiles[:split]
            train_tiles = train_tiles[split:]

        tmp_parent = Path(weights_dir).resolve().parent
        tmp_parent.mkdir(parents=True, exist_ok=True)
        data_dir = Path(tempfile.mkdtemp(prefix="yolo_data_", dir=tmp_parent))
        try:
            _write_split(data_dir, "train", train_tiles, points, bbox_px, imgsz)
            _write_split(data_dir, "val",   val_tiles,   points, bbox_px, imgsz)
            yaml_path = _write_yaml(data_dir)
            self._model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                project=str(Path(weights_dir).resolve()),
                name=name,
            )
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

        best = Path(weights_dir) / name / "weights" / "best.pt"
        return best

    def predict(self, tile: Tile, conf: float = 0.25) -> list[Point]:
        """Run inference on a tile; returns detected lamp post positions in BNG."""
        results = self._model(tile.img_pil, conf=conf, verbose=False)
        left, bottom, right, top = tile.bbox_bng
        out: list[Point] = []
        for result in results:
            for box in result.boxes:
                cx_n, cy_n = box.xywhn[0][:2].tolist()
                x = left + cx_n * (right - left)
                y = top  - cy_n * (top - bottom)
                out.append(Point(x=x, y=y, system=System.BNG))
        return out


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_split(
    data_dir: Path,
    split: str,
    tiles: list[Tile],
    points: list[Point],
    bbox_px: int,
    imgsz: int,
) -> None:
    img_dir = data_dir / "images" / split
    lbl_dir = data_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for tile in tiles:
        img = tile.img_data
        src_h, src_w = img.shape[:2]
        left, bottom, right, top = tile.bbox_bng
        stem = Path(tile.path).stem

        # Pre-resize so YOLO doesn't repeat this work every batch
        img_resized = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(img_dir / f"{stem}.png"), img_resized)

        # Free cached image immediately — don't accumulate 1028 × 9 MB in RAM
        tile._img = None
        tile._img_pil = None

        tile_points = [p for p in points if tile.is_inside(p)]
        with open(lbl_dir / f"{stem}.txt", "w") as f:
            for p in tile_points:
                cx = (p.x - left) / (right - left)
                cy = (top - p.y) / (top - bottom)
                f.write(f"0 {cx:.6f} {cy:.6f} {bbox_px/src_w:.6f} {bbox_px/src_h:.6f}\n")


def _write_yaml(data_dir: Path) -> Path:
    yaml_path = data_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(
            {
                "path":  str(data_dir),
                "train": "images/train",
                "val":   "images/val",
                "names": {0: "lamppost"},
            },
            f,
        )
    return yaml_path
