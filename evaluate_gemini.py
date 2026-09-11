"""Evaluate the Gemini full-tile lamp post detector on a sample of tiles."""
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_utils.gpkg.gpkg_loader import load_points
from data_utils.models import Point, Tile
from evaluate_yolo_crop import EvalResult, load_all_tiles
from prediction_models.gemini._shared import CostTracker
from prediction_models.gemini.gemini_model import GeminiModel

_MATCH_DIST_M = 5.0
_SAMPLE_FRAC  = 0.01   # fraction of all tiles to evaluate (tweak as needed)
_SEED         = 42
_SCALE        = 1.0    # image scale factor sent to Gemini (e.g. 0.5 = half size)
_MAX_WORKERS  = 5      # parallel Gemini requests


def _process_tile(
    args: tuple[int, int, Tile, list[Point], GeminiModel, CostTracker]
) -> tuple[int, int, int, int, bool, int]:
    """Returns (tp, fp, fn, fp_empty, is_empty, gt_count)."""
    idx, total, tile, points, model, tracker = args
    gt = [p for p in points if tile.is_inside(p)]
    preds = model.predict(tile, tracker=tracker)
    print(f"[{idx}/{total}] {tile.path}  GT={len(gt)}  preds={len(preds)}")

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


def evaluate(tiles, points, scale: float = _SCALE) -> tuple[EvalResult, CostTracker]:
    model   = GeminiModel(scale=scale)
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


if __name__ == "__main__":
    collection = load_points("data/lamp_post_annotations.gpkg")
    all_tiles  = load_all_tiles("data")

    rng = random.Random(_SEED)
    shuffled = all_tiles[:]
    rng.shuffle(shuffled)
    n_sample = max(1, int(len(shuffled) * _SAMPLE_FRAC))
    sample   = shuffled[:n_sample]

    print(f"{len(all_tiles)} total tiles → evaluating {n_sample} "
          f"({_SAMPLE_FRAC*100:.1f}%)  scale={_SCALE}  workers={_MAX_WORKERS}\n")

    result, tracker = evaluate(sample, collection.points, scale=_SCALE)

    print()
    result.print()
    print()
    tracker.print_summary()
