from __future__ import annotations

import dataclasses
from typing import Final
from typing import Literal
from typing import NamedTuple

import numpy as np
import scipy.ndimage
import scipy.spatial
import skimage
from loguru import logger
from numpy.typing import NDArray

from .._shape import Shape


class Circle(NamedTuple):
    cx: float
    cy: float
    radius: float


def _round_bbox_to_int(
    *,
    bbox: tuple[float, float, float, float] | NDArray[np.floating],
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bbox
    return (
        int(round(float(xmin))),
        int(round(float(ymin))),
        int(round(float(xmax))),
        int(round(float(ymax))),
    )


def shape_to_xyxy_bbox(*, shape: Shape) -> NDArray[np.float32] | None:
    """Returns None only when a supported shape is mid-draw (too few points);
    raises ValueError for shape types that have no bbox interpretation.
    """
    if shape.shape_type == "circle":
        if len(shape.points) != 2:
            return None
        center, edge = shape.points
        radius = float(np.linalg.norm(edge - center))
        return np.array(
            [
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ],
            dtype=np.float32,
        )
    minimum_points_by_shape_type = {
        "rectangle": 2,
        "mask": 2,
        "polygon": 3,
        "oriented_rectangle": 4,
    }
    if shape.shape_type not in minimum_points_by_shape_type:
        raise ValueError(f"Unsupported shape_type: {shape.shape_type!r}")
    if len(shape.points) < minimum_points_by_shape_type[shape.shape_type]:
        return None
    xmin, ymin = shape.points.min(axis=0)
    xmax, ymax = shape.points.max(axis=0)
    return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)


def compute_circle_from_mask(mask: NDArray[np.bool_]) -> Circle | None:
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    # Area-equivalent radius: matches the mask's pixel area, not its extent.
    # For elongated or sparse masks the resulting circle may be smaller than
    # the tightest enclosing one.
    return Circle(
        cx=float(xs.mean()),
        cy=float(ys.mean()),
        radius=float(np.sqrt(mask.sum() / np.pi)),
    )


def compute_oriented_rectangle_from_mask(
    mask: NDArray[np.bool_],
) -> NDArray[np.float32] | None:
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return None
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    try:
        # Qhull returns 2D hull vertices in CCW order, which the rotating
        # calipers loop below relies on for the right-handed perpendicular.
        hull_indices = scipy.spatial.ConvexHull(points=points).vertices
    except scipy.spatial.QhullError:
        # All pixels are collinear, so no rectangle can be fit; let callers
        # fall back to the axis-aligned bbox.
        return None
    return _min_area_rect(hull=points[hull_indices]).astype(np.float32)


def _min_area_rect(hull: NDArray[np.float64]) -> NDArray[np.float64]:
    # Rotating calipers: the minimum-area enclosing rectangle must have one
    # side flush with an edge of the convex hull. Try each hull edge as the
    # rect orientation and keep the smallest-area candidate.
    best_area = float("inf")
    best_corners: NDArray[np.float64] | None = None
    n = len(hull)
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        length = float(np.linalg.norm(edge))
        if length == 0:
            continue
        u = edge / length
        perp = np.array([-u[1], u[0]])
        u_coords = hull @ u
        p_coords = hull @ perp
        u_min, u_max = float(u_coords.min()), float(u_coords.max())
        p_min, p_max = float(p_coords.min()), float(p_coords.max())
        u_extent = u_max - u_min
        p_extent = p_max - p_min
        area = u_extent * p_extent
        if area >= best_area:
            continue
        best_area = area
        center = (u_min + u_max) / 2 * u + (p_min + p_max) / 2 * perp
        if u_extent >= p_extent:
            long_axis, half_long, half_short = u, u_extent / 2, p_extent / 2
        else:
            long_axis, half_long, half_short = perp, p_extent / 2, u_extent / 2
        # Pin the long axis to the right half-plane (or to the lower
        # half-plane when it is exactly vertical) so the corner sequence is
        # platform-independent.
        if long_axis[0] < 0 or (long_axis[0] == 0 and long_axis[1] < 0):
            long_axis = -long_axis
        # Right-handed perpendicular yields a deterministic corner traversal:
        # p0 → p1 along the long axis, then p1 → p2 along the short axis.
        short_axis = np.array([-long_axis[1], long_axis[0]])
        best_corners = np.array(
            [
                center - long_axis * half_long - short_axis * half_short,
                center + long_axis * half_long - short_axis * half_short,
                center + long_axis * half_long + short_axis * half_short,
                center - long_axis * half_long + short_axis * half_short,
            ]
        )
    # Callers filter hulls with fewer than three distinct points, so the loop
    # above always finds at least one positive-length edge.
    assert best_corners is not None
    return best_corners


def _get_contour_length(contour: NDArray[np.float32]) -> float:
    contour_start: NDArray[np.float32] = contour
    contour_end: NDArray[np.float32] = np.r_[contour[1:], contour[0:1]]
    return np.linalg.norm(contour_end - contour_start, axis=1).sum()


RefineMethod = Literal["none", "chan_vese", "geodesic"]


@dataclasses.dataclass(frozen=True)
class PolygonSimplification:
    # Ramer-Douglas-Peucker tolerance: max distance (pixels) a dropped vertex
    # may lie from the simplified outline.
    epsilon: float = 2.0
    # Share each vertex keeps per smoothing pass; the rest is split evenly
    # between its two neighbors. 1.0 disables smoothing, lower is smoother.
    smoothing_weight: float = 0.5
    # Smoothing reach is about sqrt(iterations * (1 - weight)) pixels; the
    # default (~4px) irons out SAM's blocky low-res mask steps.
    smoothing_iterations: int = 30
    # Closing radius (pixels) that fuses mask fragments split by gaps or thin
    # necks into one region. 0 disables merging.
    merge_radius: int = 5
    # Level-set refinement that snaps the mask boundary to the image. It may
    # only move the boundary within refine_band pixels of the original mask,
    # so it cannot leak into neighboring structures of similar intensity.
    refine_method: RefineMethod = "geodesic"
    refine_iterations: int = 15
    refine_band: int = 8
    # Final step: move each vertex along its normal onto the strongest image
    # edge within this many pixels, undoing the drift smoothing/RDP introduce.
    # 0 disables snapping.
    snap_radius: float = 3.0
    # Simplify Selected Polygons only: re-run SAM with each polygon's bounding
    # box as the prompt and replace the polygon with the post-processed result,
    # instead of reshaping the existing outline. AI creation ignores it.
    resegment_with_sam: bool = False

    @property
    def mask_margin(self) -> int:
        # Room around the mask so closing/refinement can grow past its crop.
        return max(self.merge_radius, self.refine_band) + 2


def _largest_component(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    labels, num = scipy.ndimage.label(mask)
    if num <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def merge_mask_fragments(mask: NDArray[np.bool_], *, radius: int) -> NDArray[np.bool_]:
    """Fuse nearby fragments into one hole-free region.

    The mask needs at least ``radius`` pixels of empty border, otherwise the
    closing is clipped at the array edge.
    """
    if radius <= 0 or not mask.any():
        return mask
    merged = skimage.morphology.binary_closing(
        mask, footprint=skimage.morphology.disk(radius)
    )
    return scipy.ndimage.binary_fill_holes(_largest_component(merged))


def refine_mask_boundary(
    mask: NDArray[np.bool_],
    gray: NDArray[np.floating],
    *,
    method: RefineMethod,
    iterations: int,
    band: int,
) -> NDArray[np.bool_]:
    """Snap the mask boundary to the image with a morphological level set."""
    if method == "none" or iterations <= 0 or band <= 0 or not mask.any():
        return mask
    if method == "chan_vese":
        # Region-based: separates the object from its surroundings by mean
        # intensity, which suits saturated objects such as metal implants.
        refined = skimage.segmentation.morphological_chan_vese(
            gray, num_iter=iterations, init_level_set=mask, smoothing=2
        )
    elif method == "geodesic":
        # Edge-based: follows strong gradients, more robust when a touching
        # neighbor has the same intensity.
        edge_map = skimage.segmentation.inverse_gaussian_gradient(
            gray, alpha=200, sigma=1.5
        )
        refined = skimage.segmentation.morphological_geodesic_active_contour(
            edge_map,
            num_iter=iterations,
            init_level_set=mask,
            smoothing=1,
            balloon=0,
            threshold=0.5,
        )
    else:
        raise ValueError(f"Unsupported refine method: {method!r}")
    footprint = skimage.morphology.disk(band)
    inner = skimage.morphology.binary_erosion(mask, footprint=footprint)
    outer = skimage.morphology.binary_dilation(mask, footprint=footprint)
    refined = (refined.astype(bool) | inner) & outer
    refined = scipy.ndimage.binary_fill_holes(_largest_component(refined))
    return refined if refined.any() else mask


def _crop_gray(
    image: NDArray[np.uint8], *, x0: int, y0: int, height: int, width: int
) -> NDArray[np.float32] | None:
    """Grayscale crop in [0, 1]; parts outside the image repeat the edge."""
    image_height, image_width = image.shape[:2]
    cy0, cy1 = max(y0, 0), min(y0 + height, image_height)
    cx0, cx1 = max(x0, 0), min(x0 + width, image_width)
    if cy0 >= cy1 or cx0 >= cx1:
        return None
    crop = image[cy0:cy1, cx0:cx1]
    gray = (crop.mean(axis=2) if crop.ndim == 3 else crop).astype(np.float32) / 255
    return np.pad(
        gray,
        ((cy0 - y0, y0 + height - cy1), (cx0 - x0, x0 + width - cx1)),
        mode="edge",
    )


def postprocess_mask(
    mask: NDArray[np.bool_],
    *,
    origin_xy: NDArray[np.floating],
    image: NDArray[np.uint8] | None,
    simplification: PolygonSimplification,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Merge fragments and refine the boundary of a mask placed at origin_xy.

    Returns the processed mask, padded by ``simplification.mask_margin``, and
    its new image-space origin.
    """
    margin = simplification.mask_margin
    padded = np.pad(mask, margin)
    origin = np.asarray(origin_xy, dtype=np.float64) - margin
    processed = merge_mask_fragments(padded, radius=simplification.merge_radius)
    if image is not None and simplification.refine_method != "none":
        gray = _crop_gray(
            image,
            x0=int(round(origin[0])),
            y0=int(round(origin[1])),
            height=padded.shape[0],
            width=padded.shape[1],
        )
        if gray is not None:
            processed = refine_mask_boundary(
                processed,
                gray,
                method=simplification.refine_method,
                iterations=simplification.refine_iterations,
                band=simplification.refine_band,
            )
    return processed, origin


def snap_vertices_to_edges(
    points: NDArray[np.floating],
    image: NDArray[np.uint8],
    *,
    radius: float,
    sigma: float = 1.0,
) -> NDArray[np.float64]:
    """Move each xy vertex along its normal onto the strongest nearby edge.

    Vertices whose best edge is weak (below half the median edge strength of
    the polygon) stay put, so flat regions do not attract them.
    """
    points = np.asarray(points, dtype=np.float64)
    if radius <= 0 or len(points) < 3:
        return points
    STEP: Final[float] = 0.25
    # Prefer the closer of two similar edges so a vertex is not pulled across
    # a thin structure onto its far side.
    DISTANCE_PENALTY: Final[float] = 0.3
    pad = int(np.ceil(radius + 3 * sigma)) + 1
    x0, y0 = np.floor(points.min(axis=0)).astype(int) - pad
    x1, y1 = np.ceil(points.max(axis=0)).astype(int) + pad
    gray = _crop_gray(image, x0=x0, y0=y0, height=y1 - y0, width=x1 - x0)
    if gray is None:
        return points
    gradient = np.hypot(
        scipy.ndimage.gaussian_filter(gray, sigma, order=(0, 1)),
        scipy.ndimage.gaussian_filter(gray, sigma, order=(1, 0)),
    )
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    offsets = np.arange(-radius, radius + STEP / 2, STEP)
    samples = (
        points[:, None, :] + offsets[None, :, None] * normal[:, None, :] - [x0, y0]
    )
    # Cubic interpolation: linear interpolation turns a step edge's gradient
    # into a flat plateau between two pixels, hiding the peak's position.
    profile = scipy.ndimage.map_coordinates(
        gradient,
        [samples[..., 1].ravel(), samples[..., 0].ravel()],
        order=3,
        mode="nearest",
    ).reshape(len(points), len(offsets))
    # Candidates are the local maxima of the gradient profile. The distance
    # penalty only ranks candidates; it must not shift a peak's position.
    center = profile[:, 1:-1]
    is_peak = (center >= profile[:, :-2]) & (center > profile[:, 2:])
    penalty = 1 - DISTANCE_PENALTY * np.abs(offsets[1:-1]) / radius
    scores = np.where(is_peak, center * penalty, -np.inf)
    best = scores.argmax(axis=1) + 1
    rows = np.arange(len(points))
    has_peak = np.isfinite(scores[rows, best - 1])
    left, mid, right = (
        profile[rows, best - 1],
        profile[rows, best],
        profile[rows, best + 1],
    )
    # Parabolic interpolation through the peak and its neighbors for a
    # sub-step estimate of the edge position.
    denominator = left - 2 * mid + right
    delta = np.where(
        np.abs(denominator) > 1e-12,
        0.5 * (left - right) / np.where(denominator == 0, 1, denominator),
        0.0,
    )
    shift = offsets[best] + np.clip(delta, -0.5, 0.5) * STEP
    strong = has_peak & (mid >= 0.5 * np.median(profile.max(axis=1)))
    snapped = points.copy()
    snapped[strong] += shift[strong, None] * normal[strong]
    return snapped


def resimplify_polygon(
    points: NDArray[np.floating],
    *,
    image: NDArray[np.uint8] | None,
    simplification: PolygonSimplification,
) -> NDArray[np.float64]:
    """Run an existing xy polygon through the same pipeline as AI output.

    Returns the input unchanged unless the result has fewer vertices, so an
    already sparse (e.g. hand-drawn) polygon is never made denser.
    """
    points = np.asarray(points, dtype=np.float64)
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int) + 1
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    rr, cc = skimage.draw.polygon(points[:, 1] - y0, points[:, 0] - x0, mask.shape)
    mask[rr, cc] = True
    if not mask.any():
        return simplify_closed_polygon(points, simplification=simplification)
    processed, origin = postprocess_mask(
        mask,
        origin_xy=np.array([x0, y0], dtype=np.float64),
        image=image,
        simplification=simplification,
    )
    polygon = compute_polygon_from_mask(processed, simplification=simplification)
    if not 3 <= len(polygon) < len(points):
        return points
    polygon = polygon.astype(np.float64) + origin
    if image is not None:
        polygon = snap_vertices_to_edges(
            polygon, image, radius=simplification.snap_radius
        )
    return polygon


def resample_closed_polygon(
    points: NDArray[np.floating], *, spacing: float = 1.0
) -> NDArray[np.float64]:
    """Resample a closed polygon to evenly spaced vertices along its perimeter."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return points
    closed = np.r_[points, points[:1]]
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    perimeter = cumulative[-1]
    num = int(perimeter / spacing)
    if num < 3:
        return points
    samples = np.linspace(0.0, perimeter, num, endpoint=False)
    return np.stack(
        [np.interp(samples, cumulative, closed[:, i]) for i in range(2)], axis=1
    )


def smooth_closed_polygon(
    points: NDArray[np.floating], *, weight: float, iterations: int
) -> NDArray[np.float64]:
    """Laplacian smoothing of a closed loop (first and last points are neighbors)."""
    smoothed = np.asarray(points, dtype=np.float64)
    if len(smoothed) < 3:
        return smoothed
    neighbor_weight = (1.0 - weight) / 2.0
    for _ in range(iterations):
        smoothed = smoothed * weight + neighbor_weight * (
            np.roll(smoothed, 1, axis=0) + np.roll(smoothed, -1, axis=0)
        )
    return smoothed


def simplify_closed_polygon(
    points: NDArray[np.floating], *, simplification: PolygonSimplification
) -> NDArray[np.float64]:
    """Smooth then RDP-simplify a closed polygon given without a repeated end point.

    Returns the input unchanged if simplification would leave fewer than 3
    vertices, so a small object is never collapsed into a degenerate shape.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return points
    # Each smoothing pass only mixes adjacent vertices, so its reach depends on
    # vertex spacing; resampling to 1px makes the iteration count mean the same
    # pixel scale for any input (about sqrt(iterations * (1 - weight)) px).
    smoothed = smooth_closed_polygon(
        resample_closed_polygon(points, spacing=1.0)
        if simplification.smoothing_iterations > 0
        else points,
        weight=simplification.smoothing_weight,
        iterations=simplification.smoothing_iterations,
    )
    # approximate_polygon keeps both ends of the input, so close the loop to
    # let RDP treat the seam like any other vertex, then drop the duplicate.
    simplified = skimage.measure.approximate_polygon(
        coords=np.r_[smoothed, smoothed[:1]],
        tolerance=simplification.epsilon,
    )[:-1]
    if len(simplified) < 3:
        return points
    return simplified


def compute_polygon_from_mask(
    mask: NDArray[np.bool_],
    simplification: PolygonSimplification | None = None,
) -> NDArray[np.float32]:
    # Pad so a region touching the image border still has a background ring to
    # close its contour against; the resulting offset is removed below.
    PAD: Final[int] = 1
    contours: NDArray[np.float32] = skimage.measure.find_contours(
        np.pad(mask, pad_width=PAD)
    )
    if len(contours) == 0:
        logger.warning("No contour found, so returning empty polygon.")
        return np.empty((0, 2), dtype=np.float32)

    contour: NDArray[np.float32] = max(contours, key=_get_contour_length)
    contour = contour - PAD  # undo the pad shift: back to image-space coords
    if simplification is None:
        POLYGON_APPROX_TOLERANCE: Final[float] = 0.004
        polygon: NDArray[np.float32] = skimage.measure.approximate_polygon(
            coords=contour,
            tolerance=np.ptp(contour, axis=0).max() * POLYGON_APPROX_TOLERANCE,
        )
        polygon = polygon[:-1]  # drop last point that is duplicate of first point
    else:
        polygon = simplify_closed_polygon(
            contour[:-1], simplification=simplification
        ).astype(np.float32)
    # np.clip is inclusive and a half-pixel boundary can reach shape - 0.5, so
    # the ceiling is mask.shape (not mask.shape - 1) to keep edge-touching
    # contours from being pulled inward by a pixel.
    polygon = np.clip(polygon, (0, 0), (mask.shape[0], mask.shape[1]))

    return polygon[:, ::-1]  # yx -> xy
