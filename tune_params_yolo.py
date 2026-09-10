"""
tune_params_yolo.py — Tune YOLO inference (conf, iou) to recall >= TARGET_RECALL.

Strategy
--------
Precompute raw detections (very low conf, NMS disabled) once per fold using that
fold's trained weights on its held-out val tiles.  Then sweep a conf×iou grid in
pure Python — no extra GPU calls — using 5-fold outer CV:

  Outer fold k:
    search: aggregate TP/FP/FN across folds {1..K} \\ {k} to find best (conf, iou)
    test:   evaluate those params on fold k's tiles with fold k's weights

Splits are reproduced from the same seed/order as evaluate.py, so folds[i]
matches exactly the val tiles that weight_paths[i] was trained without.
"""
import gc
import random
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from ultralytics import YOLO

from data_utils.gpkg.gpkg_loader import load_points
from data_utils.models import Point, Tile
from evaluate import load_all_tiles

_WEIGHTS_DIR    = "weights/yolo_20260910"
_DATA_GPKG      = "data/lamp_post_annotations.gpkg"
_MATCH_DIST_M   = 5.0
_TARGET_RECALL  = 0.90
_K              = 5
_SEED           = 42

# Grid to sweep (pure-Python post-filtering — no extra GPU calls)
_CONF_VALS = [0.005, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
_IOU_VALS  = [0.30, 0.45, 0.60]

# Precompute settings: capture every candidate box from YOLO
_RAW_CONF = 0.001   # very low — keep almost everything
_RAW_NMS  = 0.99    # effectively disables YOLO's NMS (we redo it ourselves)


# ── data structures ───────────────────────────────────────────────────────────

class _Box(NamedTuple):
    """Raw detection box in normalised image coords + BNG centroid + score."""
    x1n:   float
    y1n:   float
    x2n:   float
    y2n:   float
    score: float
    bng_x: float
    bng_y: float


# Per-fold cache: (per-tile boxes, per-tile GT points)
_FoldData = tuple[list[list[_Box]], list[list[Point]]]


@dataclass
class TuneResult:
    outer_fold: int
    conf:       float
    iou:        float
    search_p:   float
    search_r:   float
    test_p:     float
    test_r:     float
    test_f1:    float


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_tiles(tiles: list[Tile]) -> list[list[Tile]]:
    """Reproduce the exact fold split used in evaluate.py."""
    rng = random.Random(_SEED)
    shuffled = tiles[:]
    rng.shuffle(shuffled)
    fold_size = len(shuffled) // _K
    return [shuffled[i * fold_size:(i + 1) * fold_size] for i in range(_K)]


def _load_weight_paths() -> list[Path]:
    paths = sorted(Path(_WEIGHTS_DIR).glob("fold_*/weights/best.pt"))
    if len(paths) != _K:
        raise FileNotFoundError(
            f"Expected {_K} weight files under {_WEIGHTS_DIR}, "
            f"found {len(paths)}: {[str(p) for p in paths]}"
        )
    return paths


def _precompute_fold(
    model: YOLO, tiles: list[Tile], all_points: list[Point]
) -> _FoldData:
    """Run inference at _RAW_CONF/_RAW_NMS; cache all boxes + GT per tile."""
    tile_boxes: list[list[_Box]] = []
    tile_gts:   list[list[Point]] = []

    for tile in tiles:
        left, bottom, right, top = tile.bbox_bng
        results = model(tile.img_pil, conf=_RAW_CONF, iou=_RAW_NMS, verbose=False)

        boxes: list[_Box] = []
        for result in results:
            for box in result.boxes:
                x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
                score = float(box.conf[0])
                cx_n  = (x1n + x2n) / 2
                cy_n  = (y1n + y2n) / 2
                boxes.append(_Box(
                    x1n, y1n, x2n, y2n, score,
                    bng_x=left + cx_n * (right - left),
                    bng_y=top  - cy_n * (top - bottom),
                ))

        tile_boxes.append(boxes)
        tile_gts.append([p for p in all_points if tile.is_inside(p)])
        tile._img     = None
        tile._img_pil = None

    return tile_boxes, tile_gts


def _box_iou(a: _Box, b: _Box) -> float:
    ix1 = max(a.x1n, b.x1n); iy1 = max(a.y1n, b.y1n)
    ix2 = min(a.x2n, b.x2n); iy2 = min(a.y2n, b.y2n)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a.x2n - a.x1n) * (a.y2n - a.y1n)
    area_b = (b.x2n - b.x1n) * (b.y2n - b.y1n)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: list[_Box], iou_thresh: float) -> list[_Box]:
    kept: list[_Box] = []
    for box in sorted(boxes, key=lambda b: b.score, reverse=True):
        if all(_box_iou(box, k) < iou_thresh for k in kept):
            kept.append(box)
    return kept


def _match(preds: list[_Box], gt: list[Point]) -> tuple[int, int, int]:
    """Greedy nearest-neighbour matching within _MATCH_DIST_M metres."""
    matched: set[int] = set()
    tp = 0
    for pred in preds:
        best_d, best_j = _MATCH_DIST_M, -1
        for j, g in enumerate(gt):
            if j in matched:
                continue
            d = ((pred.bng_x - g.x) ** 2 + (pred.bng_y - g.y) ** 2) ** 0.5
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            matched.add(best_j)
            tp += 1
    return tp, len(preds) - tp, len(gt) - tp


def _eval_params(
    fold_datas: list[_FoldData], conf: float, iou: float
) -> tuple[float, float]:
    """Aggregate P/R across all folds in fold_datas for given (conf, iou)."""
    tp = fp = fn = 0
    for tile_boxes, tile_gts in fold_datas:
        for raw_boxes, gt in zip(tile_boxes, tile_gts):
            filtered = [b for b in raw_boxes if b.score >= conf]
            preds    = _nms(filtered, iou)
            t, f, n  = _match(preds, gt)
            tp += t; fp += f; fn += n
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r


def _pick_best(
    grid: list[tuple[float, float, float, float]]
) -> tuple[float, float, float, float]:
    """Return entry with highest P subject to R >= TARGET; fall back to max R."""
    candidates = [(c, io, p, r) for c, io, p, r in grid if r >= _TARGET_RECALL]
    if candidates:
        return max(candidates, key=lambda x: x[2])
    return max(grid, key=lambda x: x[3])


# ── main ──────────────────────────────────────────────────────────────────────

def tune_and_evaluate(
    tiles: list[Tile], points: list[Point]
) -> list[TuneResult]:
    folds        = _split_tiles(tiles)
    weight_paths = _load_weight_paths()

    print("Pre-computing raw detections for all folds …")
    all_fold_data: list[_FoldData] = []
    for i, (fold_tiles, wpath) in enumerate(zip(folds, weight_paths)):
        print(f"  Fold {i + 1}/{_K}  ({len(fold_tiles)} tiles)  "
              f"weights={wpath.parent.parent.name}")
        model     = YOLO(str(wpath))
        fold_data = _precompute_fold(model, fold_tiles, points)
        all_fold_data.append(fold_data)
        del model
        gc.collect()

    results: list[TuneResult] = []
    grid_size = len(_CONF_VALS) * len(_IOU_VALS)

    for outer in range(_K):
        print(f"\n── Outer fold {outer + 1}/{_K} ──")

        search_data = [all_fold_data[i] for i in range(_K) if i != outer]
        test_data   = [all_fold_data[outer]]

        print(f"  Grid search ({grid_size} combos) …")
        grid: list[tuple[float, float, float, float]] = []
        for conf in _CONF_VALS:
            for iou in _IOU_VALS:
                p, r = _eval_params(search_data, conf, iou)
                grid.append((conf, iou, p, r))
                print(f"    conf={conf:.3f}  iou={iou:.2f}  P={p:.3f}  R={r:.3f}")

        best_conf, best_iou, search_p, search_r = _pick_best(grid)
        flag = "" if search_r >= _TARGET_RECALL else "  ⚠ recall target not met on search set"
        print(
            f"  → best  conf={best_conf:.3f}  iou={best_iou:.2f}  "
            f"P={search_p:.3f}  R={search_r:.3f}{flag}"
        )

        test_p, test_r = _eval_params(test_data, best_conf, best_iou)
        test_f1 = 2 * test_p * test_r / (test_p + test_r) if (test_p + test_r) else 0.0
        print(f"  Test:  P={test_p:.3f}  R={test_r:.3f}  F1={test_f1:.3f}")

        results.append(TuneResult(
            outer_fold=outer + 1,
            conf=best_conf, iou=best_iou,
            search_p=search_p, search_r=search_r,
            test_p=test_p, test_r=test_r, test_f1=test_f1,
        ))

    return results


def print_summary(results: list[TuneResult]) -> None:
    header = (
        f"{'Fold':>5}  {'conf':>6}  {'iou':>5}  "
        f"{'srch-P':>7}  {'srch-R':>7}  "
        f"{'test-P':>7}  {'test-R':>7}  {'test-F1':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.outer_fold:>5}  {r.conf:>6.3f}  {r.iou:>5.2f}  "
            f"{r.search_p:>7.3f}  {r.search_r:>7.3f}  "
            f"{r.test_p:>7.3f}  {r.test_r:>7.3f}  {r.test_f1:>8.3f}"
        )
    print("-" * len(header))
    n = len(results)
    mean_p  = sum(r.test_p  for r in results) / n
    mean_r  = sum(r.test_r  for r in results) / n
    mean_f1 = sum(r.test_f1 for r in results) / n
    print(
        f"{'mean':>5}  {'':>6}  {'':>5}  "
        f"{'':>7}  {'':>7}  "
        f"{mean_p:>7.3f}  {mean_r:>7.3f}  {mean_f1:>8.3f}"
    )


if __name__ == "__main__":
    collection = load_points(_DATA_GPKG)
    tiles      = load_all_tiles("data")
    print(f"Loaded {len(tiles)} tiles, {len(collection.points)} lamp posts\n")

    results = tune_and_evaluate(tiles, collection.points)
    print()
    print_summary(results)
