"""Grid-search scale factors for the Gemini full-tile detector.

Runs evaluate_gemini.evaluate() for each scale in _SCALES with _FOLDS
independent random samples per scale, appending every fold to results_gemini.csv.
Prints a summary table at the end.
"""
import random

from evaluate_gemini import (
    _SAMPLE_FRAC, _SEED, _MAX_WORKERS,
    _append_csv, _print_table, evaluate,
)
from evaluate_yolo_crop import EvalResult, load_all_tiles
from data_utils.gpkg.gpkg_loader import load_points
from prediction_models.gemini._shared import CostTracker

_SCALES = [0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6]
_FOLDS  = 1   # folds per scale; bump up for variance estimates

if __name__ == "__main__":
    collection = load_points("data/lamp_post_annotations.gpkg")
    all_tiles  = load_all_tiles("data")
    n_sample   = max(1, int(len(all_tiles) * _SAMPLE_FRAC))

    print(
        f"{len(all_tiles)} total tiles  {_SAMPLE_FRAC*100:.1f}% → {n_sample}/fold  "
        f"{_FOLDS} fold(s) × {len(_SCALES)} scales  workers={_MAX_WORKERS}\n"
    )

    all_results: list[tuple[float, EvalResult, CostTracker]] = []

    for scale in _SCALES:
        print(f"\n{'='*60}")
        print(f"  scale={scale}")
        print(f"{'='*60}")
        for fold in range(_FOLDS):
            seed = _SEED + fold
            rng  = random.Random(seed)
            shuffled = all_tiles[:]
            rng.shuffle(shuffled)
            sample = shuffled[:n_sample]
            print(f"  ── fold {fold+1}/{_FOLDS}  (seed={seed}) ──")
            result, tracker = evaluate(sample, collection.points, scale=scale)
            _append_csv(fold + 1, result, tracker, scale=scale)
            all_results.append((scale, result, tracker))

    # Summary table
    print("\n" + "=" * 75)
    header = (f"{'Scale':>6}  {'Tiles':>5}  {'GT':>5}  {'TP':>5}  {'FP':>5}  {'FN':>4}  "
              f"{'P':>6}  {'R':>6}  {'F1':>6}  {'Cost':>9}")
    print(header)
    print("-" * len(header))
    total_cost = 0.0
    for scale, res, tracker in all_results:
        print(
            f"{scale:>6.2f}  {res.n_tiles:>5}  {res.n_gt:>5}  {res.tp:>5}  {res.fp:>5}  "
            f"{res.fn:>4}  {res.precision:>6.3f}  {res.recall:>6.3f}  {res.f1:>6.3f}  "
            f"${tracker.cost_usd:>8.4f}"
        )
        total_cost += tracker.cost_usd
    print("-" * len(header))
    print(f"{'total':>6}  {'':>5}  {'':>5}  {'':>5}  {'':>5}  {'':>4}  "
          f"{'':>6}  {'':>6}  {'':>6}  ${total_cost:>8.4f}")
