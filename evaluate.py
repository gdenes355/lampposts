"""5-fold cross-validation for the YOLO lamp post detector."""
import os
import random
from dataclasses import dataclass, field

from data_utils.gpkg.gpkg_loader import load_points
from data_utils.models import Point, Tile
from data_utils.tile.tile_loader import load_tile
from prediction_models.yolo.yolo_model import YoloModel


@dataclass
class FoldResult:
    fold: int
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float


@dataclass
class EvalReport:
    folds: list[FoldResult] = field(default_factory=list)

    def print_summary(self) -> None:
        header = f"{'Fold':>5}  {'P':>6}  {'R':>6}  {'F1':>6}  {'mAP50':>7}  {'mAP50-95':>9}"
        print(header)
        print("-" * len(header))
        for f in self.folds:
            print(
                f"{f.fold:>5}  {f.precision:>6.3f}  {f.recall:>6.3f}  "
                f"{f.f1:>6.3f}  {f.map50:>7.3f}  {f.map50_95:>9.3f}"
            )
        print("-" * len(header))
        means = {
            "P":        sum(f.precision for f in self.folds) / len(self.folds),
            "R":        sum(f.recall    for f in self.folds) / len(self.folds),
            "F1":       sum(f.f1        for f in self.folds) / len(self.folds),
            "mAP50":    sum(f.map50     for f in self.folds) / len(self.folds),
            "mAP50-95": sum(f.map50_95  for f in self.folds) / len(self.folds),
        }
        print(
            f"{'mean':>5}  {means['P']:>6.3f}  {means['R']:>6.3f}  "
            f"{means['F1']:>6.3f}  {means['mAP50']:>7.3f}  {means['mAP50-95']:>9.3f}"
        )


def kfold_evaluate(
    tiles: list[Tile],
    points: list[Point],
    k: int = 5,
    seed: int = 42,
    epochs: int = 50,
    imgsz: int = 640,
) -> EvalReport:
    """Run k-fold cross-validation; splits by tile so no tile leaks across folds."""
    rng = random.Random(seed)
    shuffled = tiles[:]
    rng.shuffle(shuffled)

    fold_size = len(shuffled) // k
    report = EvalReport()

    for fold in range(k):
        val_tiles   = shuffled[fold * fold_size : (fold + 1) * fold_size]
        train_tiles = shuffled[: fold * fold_size] + shuffled[(fold + 1) * fold_size :]

        print(f"\n── Fold {fold + 1}/{k}  (train={len(train_tiles)}, val={len(val_tiles)}) ──")

        model = YoloModel()
        model.train(
            train_tiles=train_tiles,
            points=points,
            val_tiles=val_tiles,
            epochs=epochs,
            imgsz=imgsz,
            name=f"fold_{fold + 1}",
        )

        # Ultralytics stores validation metrics in the trainer after training
        metrics = model._model.trainer.metrics
        p   = metrics.get("metrics/precision(B)", 0.0)
        r   = metrics.get("metrics/recall(B)",    0.0)
        f1  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        m50 = metrics.get("metrics/mAP50(B)",     0.0)
        m95 = metrics.get("metrics/mAP50-95(B)",  0.0)

        result = FoldResult(fold=fold + 1, precision=p, recall=r, f1=f1, map50=m50, map50_95=m95)
        report.folds.append(result)
        print(f"  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  mAP50={m50:.3f}  mAP50-95={m95:.3f}")

    return report


def load_all_tiles(data_dir: str = "data") -> list[Tile]:
    """Load every tile from lampposts/ and no_lampposts/ subdirectories."""
    tiles: list[Tile] = []
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

    annotated = sum(1 for t in tiles if any(t.is_inside(p) for p in collection.points))
    print(f"Found {len(tiles)} tiles total ({annotated} with annotations), {len(collection.points)} lamp posts")

    report = kfold_evaluate(tiles, collection.points)
    print()
    report.print_summary()
