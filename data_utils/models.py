import numpy as np
import rasterio
from enum import StrEnum
from PIL import Image
from pydantic import BaseModel, PrivateAttr


class System(StrEnum):
    BNG = "BNG"


class Point(BaseModel):
    x: float
    y: float
    system: System

    _tile: "Tile | None" = PrivateAttr(default=None)

    @property
    def tile(self) -> "Tile | None":
        if self._tile is None:
            from data_utils.tile.tile_loader import find_tile
            self._tile = find_tile(self.x, self.y)
        return self._tile

    def __str__(self) -> str:
        return f"Point(x={self.x}, y={self.y}, system={self.system})"

    def __repr__(self) -> str:
        return self.__str__()


class LampPostCollection(BaseModel):
    points: list[Point]


class Tile(BaseModel):
    path: str
    bbox_bng: tuple[int, int, int, int]  # left, bottom, right, top (BNG metres)

    _img: np.ndarray | None = PrivateAttr(default=None)
    _img_pil: Image.Image | None = PrivateAttr(default=None)

    @property
    def img_data(self) -> np.ndarray:
        if self._img is None:
            with rasterio.open(self.path) as ds:
                data = ds.read()          # (bands, H, W)
                img = np.squeeze(data, axis=0) if ds.count == 1 else np.moveaxis(data, 0, -1)
            if img.max() <= 1:
                img = (255 - img * 255).astype(np.uint8)
            self._img = img
        return self._img

    @property
    def img_pil(self) -> Image.Image:
        if self._img_pil is None:
            self._img_pil = Image.fromarray(self.img_data)
        return self._img_pil

    def is_inside(self, point: Point) -> bool:
        assert point.system == System.BNG, f"Expected BNG, got {point.system}"
        left, bottom, right, top = self.bbox_bng
        return left <= point.x < right and bottom <= point.y < top

    def __str__(self) -> str:
        loaded = self._img is not None
        return f"Tile(path={self.path}, bbox_bng={self.bbox_bng}, img_loaded={loaded})"

    def __repr__(self) -> str:
        return self.__str__()
