import cv2
import numpy as np
from collections import defaultdict

from data_utils.models import Tile, LampPostCollection, Point
from data_utils.tile.tile_loader import find_tile, load_tile
from data_utils.gpkg.gpkg_loader import load_points

def show_tile_annotations(tile: Tile, collection: LampPostCollection) -> None:
    """Draw red circles on tile image at lamppost positions and display in a window."""
    points_inside = [p for p in collection.points if tile.is_inside(p)]

    img = tile.img_data.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    left, bottom, right, top = tile.bbox_bng
    h, w = img.shape[:2]

    for point in points_inside:
        px = int((point.x - left) / (right - left) * w)
        py = int((top - point.y) / (top - bottom) * h)
        cv2.circle(img, (px, py), 15, (0, 0, 255), 2)

    cv2.imshow(f"Tile {tile.path}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # load lamp post collection
    lamp_post_collection = load_points("data/lamp_post_annotations.gpkg")
    # group tiles by path
    points_by_tile_path: dict[str, list[Point]] = defaultdict(list)
    for p in lamp_post_collection.points:
        if p.tile is None:
            continue
        points_by_tile_path[p.tile.path].append(p)
    # find tile with most lamp posts
    tile_path_with_most_lamp_posts = max(points_by_tile_path, key=lambda x: len(points_by_tile_path[x]))
    tile = load_tile(tile_path_with_most_lamp_posts)
    
    show_tile_annotations(tile, lamp_post_collection)
    print(tile.path)
