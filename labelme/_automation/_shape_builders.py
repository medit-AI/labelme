from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from .._shape import Shape
from ._geometry import Circle
from ._geometry import PolygonSimplification
from ._geometry import _round_bbox_to_int
from ._geometry import compute_circle_from_mask
from ._geometry import compute_oriented_rectangle_from_mask
from ._geometry import compute_polygon_from_mask
from ._geometry import postprocess_mask
from ._geometry import snap_vertices_to_edges
from ._types import AiOutputFormat


@dataclass
class Detection:
    bbox: tuple[float, float, float, float] | None = None
    mask: NDArray[np.bool_] | None = None
    label: str | None = None
    description: str | None = None
    score: float = 0.0


def _build_shape(
    shape_type: AiOutputFormat,
    points: ArrayLike,
    *,
    mask: NDArray[np.bool_] | None = None,
    label: str | None = None,
    description: str | None = None,
) -> Shape:
    return Shape(
        label=label,
        shape_type=shape_type,
        mask=mask,
        description=description,
        points=np.asarray(points, dtype=np.float64),
        closed=True,
    )


def _shape_from_detection(
    detection: Detection,
    shape_type: AiOutputFormat,
    polygon_simplification: PolygonSimplification | None = None,
    image: NDArray[np.uint8] | None = None,
) -> Shape | None:
    if shape_type == "rectangle":
        if detection.bbox is None:
            return None
        xmin, ymin, xmax, ymax = detection.bbox
        return _build_shape(
            shape_type="rectangle",
            points=[[xmin, ymin], [xmax, ymax]],
            label=detection.label,
            description=detection.description,
        )
    if shape_type == "polygon":
        if detection.mask is None:
            return None
        mask = detection.mask
        origin = (
            np.array([detection.bbox[0], detection.bbox[1]], dtype=np.float64)
            if detection.bbox is not None
            else np.zeros(2, dtype=np.float64)
        )
        if polygon_simplification is not None:
            mask, origin = postprocess_mask(
                mask,
                origin_xy=origin,
                image=image,
                simplification=polygon_simplification,
            )
        polygon = (
            compute_polygon_from_mask(mask=mask, simplification=polygon_simplification)
            + origin
        )
        if image is not None:
            if polygon_simplification is not None:
                polygon = snap_vertices_to_edges(
                    polygon, image, radius=polygon_simplification.snap_radius
                )
            # Refinement and snapping may move vertices past the image edge.
            polygon = np.clip(polygon, (0, 0), (image.shape[1], image.shape[0]))
        if len(polygon) < 3:
            return None
        return _build_shape(
            shape_type="polygon",
            points=polygon,
            label=detection.label,
            description=detection.description,
        )
    if shape_type == "mask":
        if detection.bbox is None or detection.mask is None:
            return None
        if not detection.mask.any():
            return None
        xmin, ymin, xmax, ymax = _round_bbox_to_int(bbox=detection.bbox)
        return _build_shape(
            shape_type="mask",
            points=[[xmin, ymin], [xmax, ymax]],
            mask=detection.mask,
            label=detection.label,
            description=detection.description,
        )
    if shape_type == "circle":
        circle = _circle_for_detection(detection=detection)
        if circle is None:
            return None
        return _build_shape(
            shape_type="circle",
            points=[
                [circle.cx, circle.cy],
                [circle.cx + circle.radius, circle.cy],
            ],
            label=detection.label,
            description=detection.description,
        )
    if shape_type == "oriented_rectangle":
        corners = _oriented_rectangle_for_detection(detection=detection)
        if corners is None:
            return None
        return _build_shape(
            shape_type="oriented_rectangle",
            points=corners,
            label=detection.label,
            description=detection.description,
        )
    raise ValueError(f"Unsupported shape_type: {shape_type!r}")


def _oriented_rectangle_for_detection(
    detection: Detection,
) -> NDArray[np.float32] | None:
    if detection.mask is not None:
        corners = compute_oriented_rectangle_from_mask(mask=detection.mask)
        if corners is not None:
            offset_x, offset_y = (
                detection.bbox[:2] if detection.bbox is not None else (0.0, 0.0)
            )
            return corners + np.array([offset_x, offset_y], dtype=np.float32)
    if detection.bbox is not None:
        xmin, ymin, xmax, ymax = detection.bbox
        return np.array(
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
            dtype=np.float32,
        )
    return None


def _circle_for_detection(detection: Detection) -> Circle | None:
    if detection.mask is not None:
        circle = compute_circle_from_mask(mask=detection.mask)
        if circle is not None:
            offset_x, offset_y = (
                detection.bbox[:2] if detection.bbox is not None else (0.0, 0.0)
            )
            return Circle(
                cx=circle.cx + offset_x,
                cy=circle.cy + offset_y,
                radius=circle.radius,
            )
    if detection.bbox is not None:
        # Inscribed in bbox when no usable mask is available.
        xmin, ymin, xmax, ymax = detection.bbox
        radius = min(xmax - xmin, ymax - ymin) / 2
        if radius > 0:
            return Circle(cx=(xmin + xmax) / 2, cy=(ymin + ymax) / 2, radius=radius)
    return None


# Output formats that drop a bbox-only detection (the builder returns None when
# the mask is absent). Derived from _shape_from_detection so it cannot drift; the
# probe mirrors the runtime warning condition (a box but no mask).
MASK_REQUIRED_SHAPE_TYPES: Final[frozenset[AiOutputFormat]] = frozenset(
    shape_type
    for shape_type in typing.get_args(AiOutputFormat)
    if _shape_from_detection(
        detection=Detection(bbox=(0, 0, 1, 1), mask=None), shape_type=shape_type
    )
    is None
)


def shapes_from_detections(
    detections: list[Detection],
    shape_type: AiOutputFormat,
    polygon_simplification: PolygonSimplification | None = None,
    image: NDArray[np.uint8] | None = None,
) -> list[Shape]:
    shapes: list[Shape] = []
    for detection in detections:
        shape = _shape_from_detection(
            detection=detection,
            shape_type=shape_type,
            polygon_simplification=polygon_simplification,
            image=image,
        )
        if shape is not None:
            shapes.append(shape)
    return shapes
