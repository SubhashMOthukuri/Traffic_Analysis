"""Geometric utility functions for bounding box and point calculations."""

from __future__ import annotations
from typing import Tuple
import math


Point = Tuple[int, int]
Box = Tuple[int, int, int, int]


def midpoint(box: Box) -> Point:
    """Return integer midpoint of (x1, y1, x2, y2) box."""
    x1, y1, x2, y2 = box
    mx = int((x1 + x2) // 2)
    my = int((y1 + y2) // 2)
    return (mx, my)


def pixel_distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])