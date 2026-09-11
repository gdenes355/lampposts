"""Evaluate the Gemini sliding-window crop detector on a sample of tiles."""
import csv
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from data_utils.gpkg.gpkg_loader import load_points
from data_utils.models import Point, Tile
from evaluate_yolo_crop import EvalResult, load_all_tiles
from prediction_models.gemini._shared import CostTracker
from prediction_models.gemini.gemini_crop_model import GeminiCropModel

_MATCH_DIST_M = 5.0
_SAMPLE_FRAC  = 0.01   # fraction of all tiles per fold (tweak as needed)
_SEED         = 42
_CROP_PX      = 500    # crop window size in pixels
_MAX_WORKERS  = 3      # parallel Gemini requests (each tile fires ~9 crop calls)
_FOLDS        = 3      # number of independent random samples to run
_CSV_PATH     = Path("results_gemini_crop.csv")


def _process_tile(
    args: tuple[int, int, Tile, list[Point], GeminiCropModel, CostTracker]
) -> tuple[int, int, int, int, bool, int]:
    """Returns (tp, fp, fn, fp_empty, is_empty, gt_count)."""
    idx, total, tile, points, model, tracker = args
    gt = [p for p in points if tile.is_inside(p)]
    preds = model.predict(tile, tracker=tracker)
    print(f"  [{idx}/{total}] {tile.path}  GT={len(gt)}  preds={len(preds)}")

    if not gt:
        return 0, len(preds), 0, len(preds), True, 0

    matched_gt   = set()
    matched_pred = set()
    for i_p, pred in enumerate(preds):
        for j, truth in enumerate(gt):
            if j in matched_gt:
                continue
            if math.dist((pred.x, pred.y), (truth.x, truth.y)) <= _MATCH_DIST_M:
                matched_gt.add(j)
                matched_pred.add(i_p)
                break

    tp = len(matched_gt)
    fp = len(preds) - len(matched_pred)
    fn = len(gt)    - len(matched_gt)
    return tp, fp, fn, 0, False, len(gt)


def evaluate(tiles, points, crop_px: int = _CROP_PX) -> tuple[EvalResult, CostTracker]:
    model   = GeminiCropModel(crop_px=crop_px)
    tracker = CostTracker()

    tp = fp = fn = fp_empty = n_empty = n_gt_total = 0
    total = len(tiles)

    args_list = [(i + 1, total, tile, points, model, tracker) for i, tile in enumerate(tiles)]

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_process_tile, args): args for args in args_list}
        for future in as_completed(futures):
            t, f, n, fpe, is_empty, gt_count = future.result()
            tp += t; fp += f; fn += n
            n_gt_total += gt_count
            if is_empty:
                n_empty  += 1
                fp_empty += fpe

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return EvalResult(
        n_tiles=total,
        n_gt=n_gt_total,
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        fp_empty_tiles=fp_empty,
        n_empty_tiles=n_empty,
    ), tracker


_CSV_FIELDS = [
    "timestamp", "fold", "crop_px", "sample_frac", "n_tiles",
    "n_gt", "tp", "fp", "fn", "precision", "recall", "f1", "cost_usd", "thinking_budget",
]


def _append_csv(fold: int, result: EvalResult, tracker: CostTracker, crop_px: int) -> None:
    from prediction_models.gemini._shared import _THINKING_BUDGET
    write_header = not _CSV_PATH.exists()
    with _CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({
            "timestamp":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fold":            fold,
            "crop_px":         crop_px,
            "sample_frac":     _SAMPLE_FRAC,
            "n_tiles":         result.n_tiles,
            "n_gt":            result.n_gt,
            "tp":              result.tp,
            "fp":              result.fp,
            "fn":              result.fn,
            "precision":       round(result.precision, 6),
            "recall":          round(result.recall, 6),
            "f1":              round(result.f1, 6),
            "cost_usd":        round(tracker.cost_usd, 6),
            "thinking_budget": _THINKING_BUDGET,
        })
    print(f"  [CSV] appended fold {fold} → {_CSV_PATH}")


def _print_table(fold_results: list[tuple[EvalResult, CostTracker]]) -> None:
    header = (f"{'Fold':>5}  {'Tiles':>5}  {'GT':>5}  {'TP':>5}  {'FP':>5}  {'FN':>4}  "
              f"{'P':>6}  {'R':>6}  {'F1':>6}  {'Cost':>9}")
    print(header)
    print("-" * len(header))
    total_tracker = CostTracker()
    for fold, (res, tracker) in enumerate(fold_results, 1):
        print(
            f"{fold:>5}  {res.n_tiles:>5}  {res.n_gt:>5}  {res.tp:>5}  {res.fp:>5}  "
            f"{res.fn:>4}  {res.precision:>6.3f}  {res.recall:>6.3f}  {res.f1:>6.3f}  "
            f"${tracker.cost_usd:>8.4f}"
        )
        total_tracker.in_tokens    += tracker.in_tokens
        total_tracker.out_tokens   += tracker.out_tokens
        total_tracker.think_tokens += tracker.think_tokens
        total_tracker.calls        += tracker.calls
        total_tracker.errors       += tracker.errors
    print("-" * len(header))
    n = len(fold_results)
    mean_p  = sum(r.precision for r, _ in fold_results) / n
    mean_r  = sum(r.recall    for r, _ in fold_results) / n
    mean_f1 = sum(r.f1        for r, _ in fold_results) / n
    print(
        f"{'mean':>5}  {'':>5}  {'':>5}  {'':>5}  {'':>5}  {'':>4}  "
        f"{mean_p:>6.3f}  {mean_r:>6.3f}  {mean_f1:>6.3f}  "
        f"${total_tracker.cost_usd:>8.4f}"
    )
    print(
        f"\nTotal API calls: {total_tracker.calls}  errors: {total_tracker.errors}  "
        f"tokens in: {total_tracker.in_tokens:,}  out: {total_tracker.out_tokens:,}  "
        f"think: {total_tracker.think_tokens:,}  total cost: ${total_tracker.cost_usd:.4f}"
    )


if __name__ == "__main__":
    collection = load_points("data/lamp_post_annotations.gpkg")
    all_tiles  = load_all_tiles("data")
    n_sample   = max(1, int(len(all_tiles) * _SAMPLE_FRAC))

    print(f"{len(all_tiles)} total tiles  {_SAMPLE_FRAC*100:.1f}% sample → "
          f"{n_sample} tiles/fold  {_FOLDS} folds  crop_px={_CROP_PX}  workers={_MAX_WORKERS}\n")

    fold_results: list[tuple[EvalResult, CostTracker]] = []
    for fold in range(_FOLDS):
        seed = _SEED + fold          # different sample each fold
        rng  = random.Random(seed)
        shuffled = all_tiles[:]
        rng.shuffle(shuffled)
        sample = shuffled[:n_sample]
        print(f"── Fold {fold + 1}/{_FOLDS}  (seed={seed}) ──")
        result, tracker = evaluate(sample, collection.points, crop_px=_CROP_PX)
        fold_results.append((result, tracker))
        _append_csv(fold + 1, result, tracker, crop_px=_CROP_PX)

    print("\n" + "=" * 70)
    _print_table(fold_results)
