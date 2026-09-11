"""Gemini full-tile lamp post detector.

Scales the tile image by `scale` before sending to the API.  No training step.
"""
from PIL import Image

from data_utils.models import Point, System, Tile
from prediction_models.gemini._shared import (
    GENERATE_CONFIG, PROMPT, CostTracker, LampPostResponse,
    _GEMINI_MODEL, has_l, make_client,
)


class GeminiModel:
    def __init__(self, scale: float = 1.0, gemini_model: str = _GEMINI_MODEL):
        self._scale  = scale
        self._model  = gemini_model
        self._client = make_client()

    def predict(self, tile: Tile, tracker: CostTracker | None = None) -> list[Point]:
        """Run Gemini inference on a full tile; returns BNG Points."""
        img = tile.img_pil
        if self._scale != 1.0:
            w, h = img.size
            img = img.resize(
                (max(1, int(w * self._scale)), max(1, int(h * self._scale))),
                Image.LANCZOS,
            )

        left, bottom, right, top = tile.bbox_bng
        points: list[Point] = []

        try:
            chat     = self._client.chats.create(model=self._model, config=GENERATE_CONFIG)
            response = chat.send_message([img, PROMPT])
        except Exception as exc:
            print(f"  [Gemini error] {exc}")
            if tracker:
                tracker.add_error()
            tile._img = None
            tile._img_pil = None
            return []

        if tracker:
            tracker.add(response.usage_metadata)
            usage = response.usage_metadata
            if usage:
                from prediction_models.gemini._shared import _PRICE_INPUT_PER_M, _PRICE_OUTPUT_PER_M
                think = usage.thoughts_token_count or 0
                out   = usage.candidates_token_count or 0
                cost  = (
                    (usage.prompt_token_count or 0) / 1_000_000 * _PRICE_INPUT_PER_M
                    + (out + think) / 1_000_000 * _PRICE_OUTPUT_PER_M
                )
                print(
                    f"  tokens in={usage.prompt_token_count}  "
                    f"out={out}  think={think}  "
                    f"tile_cost=${cost:.5f}"
                )

        if response.text:
            try:
                result = LampPostResponse.model_validate_json(response.text)
            except Exception:
                tile._img = None
                tile._img_pil = None
                return []

            for bbox in result.lamp_posts:
                if not has_l(bbox.label_text):
                    continue
                cx_n = (bbox.xmin + bbox.xmax) / 2 / 1000
                cy_n = (bbox.ymin + bbox.ymax) / 2 / 1000
                points.append(Point(
                    x=left + cx_n * (right - left),
                    y=top  - cy_n * (top - bottom),
                    system=System.BNG,
                ))

        tile._img = None
        tile._img_pil = None
        return points
