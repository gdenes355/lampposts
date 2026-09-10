import fiona
from shapely.geometry import shape
from typing import NamedTuple
from data_utils.models import System, Point as ModelPoint, LampPostCollection


class Point(NamedTuple):
    x: float  # BNG easting
    y: float  # BNG northing


def load_points(path: str) -> LampPostCollection:
    """Load annotation centroids from a GeoPackage file as BNG (x, y) points."""
    with fiona.open(path) as src:
        raw_points = [Point(*shape(f.geometry).centroid.coords[0]) for f in src]
    return LampPostCollection(points=[
        ModelPoint(x=point.x, y=point.y, system=System.BNG)
        for point in raw_points
    ])


if __name__ == "__main__":
    points = load_points("data/lamp_post_annotations.gpkg")
    for point in points.points:
        print(point)
