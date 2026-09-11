"""
tune_params_yolo_crop.py — Tune crop-model inference (conf, nms_m) to recall >= TARGET_RECALL.

Strategy
--------
Pre-compute raw detections from sliding-window inference once per val tile at a
very low conf threshold (YOLO's own NMS disabled).  Then sweep a conf × nms_m
grid in pure Python — no extra GPU calls.

The grid axes are:
  conf   — minimum YOLO score to keep a detection
  nms_m  — BNG-space suppression radius in metres (replaces the fixed _NMS_M=3.0)

Uses the same 80/20 split (seed=42) as evaluate_yolo_crop.py so val tiles are
identical to those used for the final EvalResult.
"""
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

from data_utils.gpkg.gpkg_loader import load_points
from data_utils.models import Point, System, Tile
from evaluate_yolo_crop import load_all_tiles

_WEIGHTS      = "weights/yolo_crop_20260911/eval_run/weights/best.pt"
_DATA_GPKG    = "data/lamp_post_annotations.gpkg"
_MATCH_DIST_M = 5.0
_TARGET_RECALL = 1.0
_SEED         = 42

_CROP_PX  = 500
_TRAIN_SZ = 640
_STRIDE   = 375

# Pre-compute settings: keep almost every box, YOLO NMS effectively off
_RAW_CONF = 0.001
_RAW_NMS  = 0.99

# Grid to sweep (pure-Python post-filtering)
_CONF_VALS  = [0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
_NMS_M_VALS = [1.0, 2.0, 3.0, 5.0, 7.0]


# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class RawDet:
    """Single detection in BNG space."""
    bng_x: float
    bng_y: float
    score: float


# per-tile: (list[RawDet], list[Point GT])
_TileData = tuple[list[RawDet], list[Point]]


@dataclass
class GridResult:
    conf:    float
    nms_m:   float
    tp: int; fp: int; fn: int
    precision: float
    recall:    float
    f1:        float


# ── sliding-window helpers ─────────────────────────────────────────────────────

def _window_positions(w: int, h: int) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    out = []
    for y0 in range(0, h, _STRIDE):
        for x0 in range(0, w, _STRIDE):
            x1 = min(x0 + _CROP_PX, w)
            y1 = min(y0 + _CROP_PX, h)
            box = (x1 - _CROP_PX, y1 - _CROP_PX, x1, y1)
            if box not in seen:
                seen.add(box)
                out.append(box)
    return out


def _precompute_tile(model: YOLO, tile: Tile, all_points: list[Point]) -> _TileData:
    img = tile.img_data
    h, w = img.shape[:2]
    left, bottom, right, top = tile.bbox_bng

    windows = _window_positions(w, h)
    crops = [
        cv2.resize(img[y0:y1, x0:x1], (_TRAIN_SZ, _TRAIN_SZ), interpolation=cv2.INTER_AREA)
        for x0, y0, x1, y1 in windows
    ]

    dets: list[RawDet] = []
    for (x0, y0, x1, y1), result in zip(windows, model(crops, conf=_RAW_CONF, iou=_RAW_NMS, verbose=False)):
        cw, ch = x1 - x0, y1 - y0
        for box in result.boxes:
            cx_n, cy_n = box.xywhn[0][:2].tolist()
            px = x0 + cx_n * cw
            py = y0 + cy_n * ch
            dets.append(RawDet(
                bng_x=left + px / w * (right - left),
                bng_y=top  - py / h * (top - bottom),
                score=float(box.conf[0]),
            ))

    gt = [p for p in all_points if tile.is_inside(p)]
    tile._img = None
    tile._img_pil = None
    return dets, gt


# ── BNG-space NMS ──────────────────────────────────────────────────────────────

def _nms(dets: list[RawDet], nms_m: float) -> list[RawDet]:
    kept: list[RawDet] = []
    for d in sorted(dets, key=lambda x: x.score, reverse=True):
        if all(math.dist((d.bng_x, d.bng_y), (k.bng_x, k.bng_y)) >= nms_m for k in kept):
            kept.append(d)
    return kept


# ── matching ───────────────────────────────────────────────────────────────────

def _match(preds: list[RawDet], gt: list[Point]) -> tuple[int, int, int]:
    matched: set[int] = set()
    tp = 0
    for pred in preds:
        best_d, best_j = _MATCH_DIST_M, -1
        for j, g in enumerate(gt):
            if j in matched:
                continue
            d = math.dist((pred.bng_x, pred.bng_y), (g.x, g.y))
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            matched.add(best_j)
            tp += 1
    return tp, len(preds) - tp, len(gt) - tp


# ── grid evaluation ────────────────────────────────────────────────────────────

def _eval_params(tile_datas: list[_TileData], conf: float, nms_m: float) -> GridResult:
    tp = fp = fn = 0
    for dets, gt in tile_datas:
        filtered = [d for d in dets if d.score >= conf]
        preds    = _nms(filtered, nms_m)
        t, f, n  = _match(preds, gt)
        tp += t; fp += f; fn += n
    p  = tp / (tp + fp) if tp + fp else 0.0
    r  = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return GridResult(conf=conf, nms_m=nms_m, tp=tp, fp=fp, fn=fn,
                      precision=p, recall=r, f1=f1)


def _pick_best(results: list[GridResult]) -> GridResult:
    candidates = [r for r in results if r.recall >= _TARGET_RECALL]
    if candidates:
        return max(candidates, key=lambda r: r.precision)
    return max(results, key=lambda r: r.recall)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    collection = load_points(_DATA_GPKG)
    tiles      = load_all_tiles("data")

    rng = random.Random(_SEED)
    shuffled = tiles[:]
    rng.shuffle(shuffled)
    split      = int(len(shuffled) * 0.8)
    val_tiles  = shuffled[split:]

    print(f"Loaded {len(tiles)} tiles, {len(collection.points)} lamp posts")
    print(f"Val set: {len(val_tiles)} tiles\n")

    print(f"Loading weights from {_WEIGHTS} …")
    model = YOLO(_WEIGHTS)

    print(f"Pre-computing raw detections for {len(val_tiles)} val tiles …")
    tile_datas: list[_TileData] = []
    for i, tile in enumerate(val_tiles):
        dets, gt = _precompute_tile(model, tile, collection.points)
        tile_datas.append((dets, gt))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(val_tiles)} tiles done")
    print(f"  {len(val_tiles)}/{len(val_tiles)} tiles done\n")

    grid_size = len(_CONF_VALS) * len(_NMS_M_VALS)
    print(f"Grid search ({grid_size} combos) …")
    header = f"{'conf':>7}  {'nms_m':>6}  {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>5}  {'FP':>5}  {'FN':>4}"
    print(header)
    print("-" * len(header))

    all_results: list[GridResult] = []
    for conf in _CONF_VALS:
        for nms_m in _NMS_M_VALS:
            r = _eval_params(tile_datas, conf, nms_m)
            all_results.append(r)
            flag = " *" if r.recall >= _TARGET_RECALL else ""
            print(f"{conf:>7.3f}  {nms_m:>6.1f}  {r.precision:>6.3f}  {r.recall:>6.3f}  "
                  f"{r.f1:>6.3f}  {r.tp:>5}  {r.fp:>5}  {r.fn:>4}{flag}")

    best = _pick_best(all_results)
    flag = "" if best.recall >= _TARGET_RECALL else "  ⚠ recall target not met"
    print("-" * len(header))
    print(f"\nBest: conf={best.conf:.3f}  nms_m={best.nms_m:.1f}  "
          f"P={best.precision:.3f}  R={best.recall:.3f}  F1={best.f1:.3f}{flag}")
    print(f"      TP={best.tp}  FP={best.fp}  FN={best.fn}")


if __name__ == "__main__":
    main()
