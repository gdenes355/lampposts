"""Evaluate the Gemini sliding-window crop detector on a sample of tiles."""
import math
import random

from data_utils.gpkg.gpkg_loader import load_points
from evaluate_yolo_crop import EvalResult, load_all_tiles
from prediction_models.gemini._shared import CostTracker
from prediction_models.gemini.gemini_crop_model import GeminiCropModel

_MATCH_DIST_M = 5.0
_SAMPLE_FRAC  = 0.01   # fraction of all tiles to evaluate (tweak as needed)
_SEED         = 42
_CROP_PX      = 500    # crop window size in pixels


def evaluate(tiles, points, crop_px: int = _CROP_PX) -> tuple[EvalResult, CostTracker]:
    model   = GeminiCropModel(crop_px=crop_px)
    tracker = CostTracker()

    tp = fp = fn = fp_empty = n_empty = n_gt_total = 0

    for i, tile in enumerate(tiles):
        gt = [p for p in points if tile.is_inside(p)]
        n_windows = _estimate_windows(tile, crop_px)
        print(f"[{i+1}/{len(tiles)}] {tile.path}  GT={len(gt)}  ~{n_windows} crops", end="  ")
        preds = model.predict(tile, tracker=tracker)
        print(f"preds={len(preds)}")
        n_gt_total += len(gt)

        if not gt:
            n_empty  += 1
            fp_empty += len(preds)
            fp       += len(preds)
            continue

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

        tp += len(matched_gt)
        fp += len(preds) - len(matched_pred)
        fn += len(gt)    - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = EvalResult(
        n_tiles=len(tiles),
        n_gt=n_gt_total,
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        fp_empty_tiles=fp_empty,
        n_empty_tiles=n_empty,
    )
    return result, tracker


def _estimate_windows(tile, crop_px: int, stride: int = 375) -> int:
    from data_utils.tile.tile_loader import load_tile
    import math
    # Tiles are 3000×3000 px; use that as a rough estimate without loading image
    w = h = 3000
    cols = math.ceil(w / stride)
    rows = math.ceil(h / stride)
    return cols * rows


if __name__ == "__main__":
    collection = load_points("data/lamp_post_annotations.gpkg")
    all_tiles  = load_all_tiles("data")

    rng = random.Random(_SEED)
    shuffled = all_tiles[:]
    rng.shuffle(shuffled)
    n_sample = max(1, int(len(shuffled) * _SAMPLE_FRAC))
    sample   = shuffled[:n_sample]

    print(f"{len(all_tiles)} total tiles → evaluating {n_sample} "
          f"({_SAMPLE_FRAC*100:.1f}%)  crop_px={_CROP_PX}\n")

    result, tracker = evaluate(sample, collection.points, crop_px=_CROP_PX)

    print()
    result.print()
    print()
    tracker.print_summary()
