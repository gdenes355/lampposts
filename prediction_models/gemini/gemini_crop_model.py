"""Gemini sliding-window lamp post detector.

Splits each tile into crop_px × crop_px windows (same grid as YoloCropModel),
sends each crop to Gemini, then merges detections with BNG-space NMS.
No training step.
"""
import math

import cv2
from PIL import Image

from data_utils.models import Point, System, Tile
from prediction_models.gemini._shared import (
    GENERATE_CONFIG, PROMPT, CostTracker, LampPostResponse,
    _GEMINI_MODEL, has_l, make_client,
)

_DEFAULT_CROP   = 500
_DEFAULT_STRIDE = 375
_NMS_M          = 3.0   # BNG metres: suppress duplicate detections closer than this


class GeminiCropModel:
    def __init__(
        self,
        crop_px: int = _DEFAULT_CROP,
        stride: int = _DEFAULT_STRIDE,
        nms_m: float = _NMS_M,
        gemini_model: str = _GEMINI_MODEL,
    ):
        self._crop_px = crop_px
        self._stride  = stride
        self._nms_m   = nms_m
        self._model   = gemini_model
        self._client  = make_client()

    def predict(self, tile: Tile, tracker: CostTracker | None = None) -> list[Point]:
        """Sliding-window Gemini inference; returns BNG Points after NMS."""
        img = tile.img_data
        h, w = img.shape[:2]
        left, bottom, right, top = tile.bbox_bng

        windows = _window_positions(w, h, self._crop_px, self._stride)
        raw: list[tuple[float, float, float]] = []   # (bng_x, bng_y, confidence=1.0)

        for x0, y0, x1, y1 in windows:
            crop_arr = img[y0:y1, x0:x1]
            crop_pil = Image.fromarray(crop_arr)
            cw, ch   = x1 - x0, y1 - y0

            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[crop_pil, PROMPT],
                    config=GENERATE_CONFIG,
                )
            except Exception as exc:
                print(f"  [Gemini error on crop ({x0},{y0})] {exc}")
                if tracker:
                    tracker.errors += 1
                continue

            if tracker:
                tracker.add(response.usage_metadata)

            if not response.text:
                continue

            try:
                result = LampPostResponse.model_validate_json(response.text)
            except Exception:
                continue

            for bbox in result.lamp_posts:
                if not has_l(bbox.label_text):
                    continue
                cx_crop = (bbox.xmin + bbox.xmax) / 2 / 1000 * cw
                cy_crop = (bbox.ymin + bbox.ymax) / 2 / 1000 * ch
                px = x0 + cx_crop
                py = y0 + cy_crop
                raw.append((
                    left + px / w * (right - left),
                    top  - py / h * (top - bottom),
                    1.0,
                ))

        tile._img     = None
        tile._img_pil = None
        return _nms(raw, self._nms_m)


# ── helpers ───────────────────────────────────────────────────────────────────

def _window_positions(w: int, h: int, size: int, stride: int) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    out = []
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(x0 + size, w)
            y1 = min(y0 + size, h)
            box = (x1 - size, y1 - size, x1, y1)
            if box not in seen:
                seen.add(box)
                out.append(box)
    return out


def _nms(dets: list[tuple[float, float, float]], dist_m: float) -> list[Point]:
    kept: list[tuple[float, float, float]] = []
    for x, y, score in sorted(dets, key=lambda d: d[2], reverse=True):
        if all(math.dist((x, y), (kx, ky)) >= dist_m for kx, ky, _ in kept):
            kept.append((x, y, score))
    return [Point(x=x, y=y, system=System.BNG) for x, y, _ in kept]
