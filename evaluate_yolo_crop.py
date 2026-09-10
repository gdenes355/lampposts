"""80/20 evaluation for the crop-based YOLO lamp post detector."""
import math
import os
import random
from dataclasses import dataclass

from data_utils.gpkg.gpkg_loader import load_points
from data_utils.models import Point, Tile
from data_utils.tile.tile_loader import load_tile
from prediction_models.yolo.yolo_crop_model import YoloCropModel

# A predicted point counts as a true positive if it is within this distance
# (in BNG metres) of a ground-truth annotation.
_MATCH_DIST_M = 5.0


@dataclass
class EvalResult:
    n_tiles: int
    n_gt: int          # total ground-truth lamp posts
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    fp_empty_tiles: int   # false positives on tiles with no annotations
    n_empty_tiles: int

    def print(self) -> None:
        print(f"Tiles evaluated : {self.n_tiles}  ({self.n_empty_tiles} with no annotations)")
        print(f"Ground truth    : {self.n_gt} lamp posts")
        print(f"TP={self.tp}  FP={self.fp}  FN={self.fn}")
        print(f"Precision : {self.precision:.3f}")
        print(f"Recall    : {self.recall:.3f}")
        print(f"F1        : {self.f1:.3f}")
        print(f"FP on empty tiles : {self.fp_empty_tiles}")


def evaluate(
    train_tiles: list[Tile],
    val_tiles: list[Tile],
    points: list[Point],
    epochs: int = 50,
    crop_px: int = 500,
    conf: float = 0.25,
    weights_dir: str = "weights/yolo_crop",
    name: str = "eval_run",
    batch: int = 64,
    neg_ratio: float = 0.0,
) -> EvalResult:
    model = YoloCropModel(crop_px=crop_px)
    print(f"Training on {len(train_tiles)} tiles, evaluating on {len(val_tiles)} tiles …")
    model.train(train_tiles, points, epochs=epochs, weights_dir=weights_dir, name=name, batch=batch, neg_ratio=neg_ratio)

    tp = fp = fn = fp_empty = n_empty = 0
    n_gt_total = 0

    for tile in val_tiles:
        gt = [p for p in points if tile.is_inside(p)]
        preds = model.predict(tile, conf=conf)
        n_gt_total += len(gt)

        if not gt:
            n_empty += 1
            fp_empty += len(preds)
            fp += len(preds)
            continue

        matched_gt  = set()
        matched_pred = set()
        for i, pred in enumerate(preds):
            for j, truth in enumerate(gt):
                if j in matched_gt:
                    continue
                dist = math.dist((pred.x, pred.y), (truth.x, truth.y))
                if dist <= _MATCH_DIST_M:
                    matched_gt.add(j)
                    matched_pred.add(i)
                    break

        tp += len(matched_gt)
        fp += len(preds) - len(matched_pred)
        fn += len(gt)    - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return EvalResult(
        n_tiles=len(val_tiles),
        n_gt=n_gt_total,
        tp=tp, fp=fp, fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        fp_empty_tiles=fp_empty,
        n_empty_tiles=n_empty,
    )


def load_all_tiles(data_dir: str = "data") -> list[Tile]:
    tiles = []
    for subdir in ("lampposts", "no_lampposts"):
        folder = os.path.join(data_dir, subdir)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".tif"):
                tiles.append(load_tile(os.path.join(folder, fname)))
    return tiles


if __name__ == "__main__":
    collection = load_points("data/lamp_post_annotations.gpkg")
    tiles = load_all_tiles("data")

    rng = random.Random(42)
    shuffled = tiles[:]
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.8)
    train_tiles = shuffled[:split]
    val_tiles   = shuffled[split:]

    print(f"{len(tiles)} total tiles → {len(train_tiles)} train / {len(val_tiles)} val")

    result = evaluate(train_tiles, val_tiles, collection.points)
    print()
    result.print()
