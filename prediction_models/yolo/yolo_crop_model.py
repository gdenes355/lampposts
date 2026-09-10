"""YOLO model trained on fixed-size crops centred on lamp post annotations."""
import random
import shutil
import tempfile
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from data_utils.models import Point, System, Tile

_BASE_WEIGHTS  = "yolo11n.pt"
_DEFAULT_CROP  = 500   # default crop size in original tile pixels
_TRAIN_SZ      = 640   # YOLO input resolution (always 640)
_DEFAULT_BBOX  = 30    # annotation box size in original pixels
_DEFAULT_STRIDE = 375  # sliding-window stride (25 % overlap with 500 px crops)
_NMS_M         = 3.0   # BNG metres: suppress duplicate detections closer than this


class YoloCropModel:
    def __init__(
        self,
        weights: str = _BASE_WEIGHTS,
        crop_px: int = _DEFAULT_CROP,
        bbox_px: int = _DEFAULT_BBOX,
        stride: int = _DEFAULT_STRIDE,
    ):
        self._model   = YOLO(weights)
        self._crop_px = crop_px
        self._bbox_px = bbox_px
        self._stride  = stride

    def train(
        self,
        train_tiles: list[Tile],
        points: list[Point],
        epochs: int = 50,
        weights_dir: str = "weights/yolo_crop",
        name: str = "train",
        seed: int = 42,
    ) -> Path:
        """Train on crops centred on each annotation (plus equal negative crops).

        Returns path to best.pt.
        """
        rng = random.Random(seed)
        val_pool = train_tiles[:]
        rng.shuffle(val_pool)
        val_tiles = val_pool[: max(1, len(val_pool) // 5)]

        tmp_parent = Path(weights_dir).resolve().parent
        tmp_parent.mkdir(parents=True, exist_ok=True)
        data_dir = Path(tempfile.mkdtemp(prefix="yolo_crop_", dir=tmp_parent))
        try:
            self._write_crops(data_dir, "train", train_tiles, points, seed)
            self._write_crops(data_dir, "val",   val_tiles,   points, seed + 1)
            _write_yaml(data_dir)
            self._model.train(
                data=str(data_dir / "dataset.yaml"),
                epochs=epochs,
                imgsz=_TRAIN_SZ,
                project=str(Path(weights_dir).resolve()),
                name=name,
            )
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)
        return Path(weights_dir) / name / "weights" / "best.pt"

    def predict(self, tile: Tile, conf: float = 0.25) -> list[Point]:
        """Sliding-window inference over a full tile; consistent with training crops.

        Each window is crop_px × crop_px, resized to 640×640, with stride overlap.
        """
        img = tile.img_data
        h, w = img.shape[:2]
        left, bottom, right, top = tile.bbox_bng

        windows = _window_positions(w, h, self._crop_px, self._stride)
        crops = [
            cv2.resize(
                img[y0:y1, x0:x1], (_TRAIN_SZ, _TRAIN_SZ), interpolation=cv2.INTER_AREA
            )
            for x0, y0, x1, y1 in windows
        ]

        raw: list[tuple[float, float, float]] = []
        for (x0, y0, x1, y1), result in zip(windows, self._model(crops, conf=conf, verbose=False)):
            cw, ch = x1 - x0, y1 - y0
            for box in result.boxes:
                cx_n, cy_n = box.xywhn[0][:2].tolist()
                px = x0 + cx_n * cw
                py = y0 + cy_n * ch
                raw.append((
                    left + px / w * (right - left),
                    top  - py / h * (top - bottom),
                    float(box.conf[0]),
                ))

        return _nms(raw, _NMS_M)

    # ── private ───────────────────────────────────────────────────────────────

    def _write_crops(
        self,
        data_dir: Path,
        split: str,
        tiles: list[Tile],
        points: list[Point],
        seed: int,
    ) -> None:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(seed)
        idx = 0

        for tile in tiles:
            img = tile.img_data
            h, w = img.shape[:2]
            left, bottom, right, top = tile.bbox_bng
            tile_pts = [p for p in points if tile.is_inside(p)]

            # positive crops — one per annotation
            for point in tile_pts:
                px = (point.x - left) / (right - left) * w
                py = (top - point.y) / (top - bottom) * h
                x0, y0, x1, y1 = _centred_window(px, py, self._crop_px, w, h)
                idx = self._save_crop(img, x0, y0, x1, y1, tile_pts, left, bottom, right, top, w, h, img_dir, lbl_dir, idx)

            # negative crops — 1:1 with positives
            n_neg  = max(2, len(tile_pts))
            generated = attempts = 0
            while generated < n_neg and attempts < 100:
                attempts += 1
                px = rng.randint(self._crop_px // 2, w - self._crop_px // 2)
                py = rng.randint(self._crop_px // 2, h - self._crop_px // 2)
                x0, y0, x1, y1 = _centred_window(px, py, self._crop_px, w, h)
                if _labels_in_window(tile_pts, x0, y0, x1, y1, left, bottom, right, top, w, h):
                    continue
                crop = cv2.resize(img[y0:y1, x0:x1], (_TRAIN_SZ, _TRAIN_SZ), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(img_dir / f"c{idx}.png"), crop)
                open(lbl_dir / f"c{idx}.txt", "w").close()
                idx += 1
                generated += 1

    def _save_crop(
        self,
        img, x0, y0, x1, y1,
        tile_pts, left, bottom, right, top, w, h,
        img_dir, lbl_dir, idx,
    ) -> int:
        crop = cv2.resize(img[y0:y1, x0:x1], (_TRAIN_SZ, _TRAIN_SZ), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(img_dir / f"c{idx}.png"), crop)
        labels = _labels_in_window(tile_pts, x0, y0, x1, y1, left, bottom, right, top, w, h)
        with open(lbl_dir / f"c{idx}.txt", "w") as f:
            for cx, cy in labels:
                f.write(f"0 {cx:.6f} {cy:.6f} {self._bbox_px/(x1-x0):.6f} {self._bbox_px/(y1-y0):.6f}\n")
        return idx + 1


# ── module-level helpers ──────────────────────────────────────────────────────

def _centred_window(px: float, py: float, size: int, w: int, h: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(int(px) - size // 2, w - size))
    y0 = max(0, min(int(py) - size // 2, h - size))
    return x0, y0, x0 + size, y0 + size


def _labels_in_window(
    tile_pts, x0, y0, x1, y1, left, bottom, right, top, w, h
) -> list[tuple[float, float]]:
    out = []
    for p in tile_pts:
        px = (p.x - left) / (right - left) * w
        py = (top - p.y) / (top - bottom) * h
        if x0 <= px < x1 and y0 <= py < y1:
            out.append(((px - x0) / (x1 - x0), (py - y0) / (y1 - y0)))
    return out


def _window_positions(w: int, h: int, size: int, stride: int) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    out = []
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(x0 + size, w)
            y1 = min(y0 + size, h)
            box = (x1 - size, y1 - size, x1, y1)  # clamp so window is always full-size
            if box not in seen:
                seen.add(box)
                out.append(box)
    return out


def _nms(detections: list[tuple[float, float, float]], dist_m: float) -> list[Point]:
    kept: list[tuple[float, float, float]] = []
    for x, y, conf in sorted(detections, key=lambda d: d[2], reverse=True):
        if all(((x - kx) ** 2 + (y - ky) ** 2) ** 0.5 >= dist_m for kx, ky, _ in kept):
            kept.append((x, y, conf))
    return [Point(x=x, y=y, system=System.BNG) for x, y, _ in kept]


def _write_yaml(data_dir: Path) -> None:
    with open(data_dir / "dataset.yaml", "w") as f:
        yaml.dump(
            {"path": str(data_dir), "train": "images/train", "val": "images/val", "names": {0: "lamppost"}},
            f,
        )
