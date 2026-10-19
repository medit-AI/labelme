from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.ndimage
import skimage.draw
from numpy.typing import NDArray

from labelme._automation._geometry import PolygonSimplification
from labelme._automation._geometry import _round_bbox_to_int
from labelme._automation._geometry import compute_circle_from_mask
from labelme._automation._geometry import compute_oriented_rectangle_from_mask
from labelme._automation._geometry import compute_polygon_from_mask
from labelme._automation._geometry import merge_mask_fragments
from labelme._automation._geometry import postprocess_mask
from labelme._automation._geometry import refine_mask_boundary
from labelme._automation._geometry import resimplify_polygon
from labelme._automation._geometry import shape_to_xyxy_bbox
from labelme._automation._geometry import simplify_closed_polygon
from labelme._automation._geometry import snap_vertices_to_edges
from labelme._automation._geometry import smooth_closed_polygon
from labelme._shape import Shape


def test_round_bbox_to_int_uses_ties_to_even_for_tuple_and_array() -> None:
    bbox = (-11.5, -10.5, 10.5, 11.5)
    expected = (-12, -10, 10, 12)

    assert _round_bbox_to_int(bbox=bbox) == expected
    assert _round_bbox_to_int(bbox=np.array(bbox, dtype=np.float32)) == expected


def test_compute_circle_from_mask_returns_none_when_empty() -> None:
    assert compute_circle_from_mask(mask=np.zeros((10, 10), dtype=bool)) is None


def test_compute_circle_from_mask_centroid_and_area_equivalent_radius() -> None:
    mask = np.zeros((11, 11), dtype=bool)
    mask[0:3, 0:3] = True

    circle = compute_circle_from_mask(mask=mask)

    assert circle is not None
    assert circle.cx == pytest.approx(1)
    assert circle.cy == pytest.approx(1)
    assert circle.radius == pytest.approx(math.sqrt(9 / math.pi))


def test_compute_oriented_rectangle_from_mask_returns_none_when_empty() -> None:
    assert (
        compute_oriented_rectangle_from_mask(mask=np.zeros((10, 10), dtype=bool))
        is None
    )


def test_compute_oriented_rectangle_from_mask_returns_none_when_single_pixel() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[3, 4] = True
    assert compute_oriented_rectangle_from_mask(mask=mask) is None


def test_compute_oriented_rectangle_from_mask_axis_aligned_wider_than_tall() -> None:
    # 21x11 mask (cols x rows): wider than tall, so the long axis is +x and the
    # corners trace (xmin, ymin) → (xmax, ymin) → (xmax, ymax) → (xmin, ymax).
    mask = np.ones((11, 21), dtype=bool)

    corners = compute_oriented_rectangle_from_mask(mask=mask)

    assert corners is not None
    expected = np.array(
        [[0, 0], [20, 0], [20, 10], [0, 10]],
        dtype=np.float32,
    )
    assert corners == pytest.approx(expected)


def test_compute_oriented_rectangle_from_mask_axis_aligned_taller_than_wide() -> None:
    # 11x21 mask (cols x rows): taller than wide, so the long axis is +y. The
    # right-handed perpendicular rotates the corner sequence one quarter turn
    # relative to the wider-than-tall case while keeping the same convention.
    mask = np.ones((21, 11), dtype=bool)

    corners = compute_oriented_rectangle_from_mask(mask=mask)

    assert corners is not None
    expected = np.array(
        [[10, 0], [10, 20], [0, 20], [0, 0]],
        dtype=np.float32,
    )
    assert corners == pytest.approx(expected)


def test_compute_oriented_rectangle_from_mask_recovers_rotation_angle(
    rotated_rectangle_mask: NDArray[np.bool_],
    rotated_rectangle_angle: float,
) -> None:
    corners = compute_oriented_rectangle_from_mask(mask=rotated_rectangle_mask)

    assert corners is not None
    edge = corners[1] - corners[0]
    recovered = math.atan2(float(edge[1]), float(edge[0]))
    assert recovered == pytest.approx(rotated_rectangle_angle, abs=math.radians(3))


def test_compute_oriented_rectangle_from_mask_returns_none_for_collinear_mask() -> None:
    # All set pixels lie on a single row, so the convex hull collapses to two
    # points and there is no rectangle to fit. Bail out so callers fall back
    # to the axis-aligned bbox.
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, :] = True
    assert compute_oriented_rectangle_from_mask(mask=mask) is None


def test_compute_oriented_rectangle_from_mask_square_mask_is_axis_aligned() -> None:
    # A square mask has a well-defined minimum-area rectangle (the bbox
    # itself), unlike the variance-based PCA approach which is ambiguous
    # under equal eigenvalues.
    mask = np.ones((11, 11), dtype=bool)

    corners = compute_oriented_rectangle_from_mask(mask=mask)

    assert corners is not None
    expected = np.array(
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        dtype=np.float32,
    )
    assert corners == pytest.approx(expected)


def test_shape_to_xyxy_bbox_circle() -> None:
    shape = Shape(
        shape_type="circle", points=np.array([(50, 40), (53, 44)], dtype=np.float64)
    )

    bbox = shape_to_xyxy_bbox(shape=shape)

    radius = math.sqrt((53 - 50) ** 2 + (44 - 40) ** 2)
    assert bbox is not None
    assert bbox.tolist() == pytest.approx(
        [50 - radius, 40 - radius, 50 + radius, 40 + radius]
    )


def test_shape_to_xyxy_bbox_polygon() -> None:
    shape = Shape(
        shape_type="polygon",
        points=np.array([(1, 2), (10, 4), (6, 12)], dtype=np.float64),
    )

    bbox = shape_to_xyxy_bbox(shape=shape)

    assert bbox is not None
    assert bbox.tolist() == pytest.approx([1, 2, 10, 12])


def test_shape_to_xyxy_bbox_returns_none_when_polygon_has_too_few_points() -> None:
    shape = Shape(
        shape_type="polygon", points=np.array([(0, 0), (10, 10)], dtype=np.float64)
    )

    assert shape_to_xyxy_bbox(shape=shape) is None


def test_shape_to_xyxy_bbox_returns_none_when_circle_has_only_center() -> None:
    shape = Shape(shape_type="circle", points=np.array([(5, 5)], dtype=np.float64))

    assert shape_to_xyxy_bbox(shape=shape) is None


def test_shape_to_xyxy_bbox_rectangle() -> None:
    shape = Shape(
        shape_type="rectangle",
        points=np.array([(10, 4), (1, 12)], dtype=np.float64),
    )

    bbox = shape_to_xyxy_bbox(shape=shape)

    assert bbox is not None
    assert bbox.tolist() == pytest.approx([1, 4, 10, 12])


def test_shape_to_xyxy_bbox_mask() -> None:
    shape = Shape(
        shape_type="mask",
        points=np.array([(20, 5), (3, 8)], dtype=np.float64),
    )

    bbox = shape_to_xyxy_bbox(shape=shape)

    assert bbox is not None
    assert bbox.tolist() == pytest.approx([3, 5, 20, 8])


def test_shape_to_xyxy_bbox_oriented_rectangle() -> None:
    shape = Shape(
        shape_type="oriented_rectangle",
        points=np.array([(1, 1), (5, 0), (6, 4), (2, 5)], dtype=np.float64),
    )

    bbox = shape_to_xyxy_bbox(shape=shape)

    assert bbox is not None
    assert bbox.tolist() == pytest.approx([1, 0, 6, 5])


def test_shape_to_xyxy_bbox_returns_none_when_rectangle_has_too_few_points() -> None:
    shape = Shape(shape_type="rectangle", points=np.array([(5, 5)], dtype=np.float64))

    assert shape_to_xyxy_bbox(shape=shape) is None


def test_shape_to_xyxy_bbox_returns_none_when_mask_has_too_few_points() -> None:
    shape = Shape(shape_type="mask", points=np.array([(5, 5)], dtype=np.float64))

    assert shape_to_xyxy_bbox(shape=shape) is None


def test_shape_to_xyxy_bbox_none_when_oriented_rectangle_has_too_few_points() -> None:
    shape = Shape(
        shape_type="oriented_rectangle",
        points=np.array([(0, 0), (5, 0), (5, 5)], dtype=np.float64),
    )

    assert shape_to_xyxy_bbox(shape=shape) is None


def test_shape_to_xyxy_bbox_raises_on_unsupported_shape_type() -> None:
    shape = Shape(shape_type="point", points=np.array([(1, 2)], dtype=np.float64))

    with pytest.raises(ValueError, match="Unsupported shape_type"):
        shape_to_xyxy_bbox(shape=shape)


def test_compute_polygon_from_mask_returns_empty_for_empty_mask() -> None:
    polygon = compute_polygon_from_mask(mask=np.zeros((5, 5), dtype=bool))

    assert polygon.shape == (0, 2)


def test_compute_polygon_from_mask_traces_rectangle_extent_in_xy_order() -> None:
    # Rows 1-3, cols 1-7 set. The result is xy-ordered, so the max extent is
    # x=7.5, y=3.5 (a yx result would swap them, which the max assertion catches).
    # The coordinates are the region's half-pixel boundary in image space:
    # x in [0.5, 7.5], y in [0.5, 3.5].
    mask = np.zeros((5, 9), dtype=bool)
    mask[1:4, 1:8] = True

    polygon = compute_polygon_from_mask(mask=mask)

    assert polygon.min(axis=0) == pytest.approx([0.5, 0.5])
    assert polygon.max(axis=0) == pytest.approx([7.5, 3.5])


def test_compute_polygon_from_mask_picks_largest_contour() -> None:
    # Two blobs; the longer-perimeter 4x6 block wins over the 2x2 one, so the
    # returned polygon spans the larger block only.  The large blob is
    # non-square so a yx/xy transposition would swap the unequal extents and
    # fail both the selection check and the axis-order check simultaneously.
    mask = np.zeros((10, 12), dtype=bool)
    mask[1:5, 1:7] = True
    mask[6:8, 6:8] = True

    polygon = compute_polygon_from_mask(mask=mask)

    assert polygon.min(axis=0) == pytest.approx([0.5, 0.5])
    assert polygon.max(axis=0) == pytest.approx([6.5, 4.5])


def test_compute_polygon_from_mask_keeps_half_pixel_boundary_at_image_edge() -> None:
    # A fully-set mask has no background border, so its contour is the outer
    # half-pixel boundary: x in [-0.5, W-0.5], y in [-0.5, H-0.5]. The near side
    # clips to 0, while the far side must clamp to mask size (W=9, H=5), not
    # size-1, so it keeps its half-pixel offset (8.5, 4.5) instead of being
    # pulled in by a pixel.
    mask = np.ones((5, 9), dtype=bool)

    polygon = compute_polygon_from_mask(mask=mask)

    assert polygon.min(axis=0) == pytest.approx([0.0, 0.0])
    assert polygon.max(axis=0) == pytest.approx([8.5, 4.5])


def test_compute_oriented_rectangle_from_mask_l_shape_is_axis_aligned() -> None:
    # An L-shape with axis-aligned arms: the minimum-area enclosing rectangle
    # is the axis-aligned bbox, regardless of how mass is distributed between
    # the arms. PCA tilts the principal axis toward the heavier arm, which is
    # the failure mode this implementation avoids.
    mask = np.zeros((20, 30), dtype=bool)
    mask[0:5, 0:30] = True
    mask[0:20, 0:5] = True

    corners = compute_oriented_rectangle_from_mask(mask=mask)

    assert corners is not None
    expected = np.array(
        [[0, 0], [29, 0], [29, 19], [0, 19]],
        dtype=np.float32,
    )
    assert corners == pytest.approx(expected)


def _ellipse_mask(size: int) -> NDArray[np.bool_]:
    yy, xx = np.mgrid[: size * 2, : size * 2]
    return np.hypot((xx - size) / (size * 0.8), (yy - size) / (size * 0.5)) < 1


def test_compute_polygon_from_mask_simplification_reduces_points() -> None:
    mask = _ellipse_mask(size=40)

    default = compute_polygon_from_mask(mask=mask)
    simplified = compute_polygon_from_mask(
        mask=mask, simplification=PolygonSimplification(epsilon=2.0)
    )

    assert 3 <= len(simplified) < len(default) / 2
    # xy order: x spans the wider (0.8) axis of the ellipse.
    extent = np.ptp(simplified, axis=0)
    assert extent[0] > extent[1]


def test_compute_polygon_from_mask_simplification_larger_epsilon_fewer_points() -> (
    None
):
    mask = _ellipse_mask(size=100)

    counts = [
        len(
            compute_polygon_from_mask(
                mask=mask, simplification=PolygonSimplification(epsilon=epsilon)
            )
        )
        for epsilon in (0.5, 2.0, 8.0)
    ]

    assert counts[0] > counts[1] > counts[2]


def test_smooth_closed_polygon_weight_one_is_identity() -> None:
    points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)

    smoothed = smooth_closed_polygon(points, weight=1.0, iterations=5)

    assert smoothed == pytest.approx(points)


def test_smooth_closed_polygon_wraps_around_seam() -> None:
    # A square is symmetric, so closed-loop smoothing must move every corner
    # by the same amount; an open-curve smoother would treat the seam corner
    # differently.
    points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)

    smoothed = smooth_closed_polygon(points, weight=0.5, iterations=1)

    assert smoothed == pytest.approx(
        np.array([[2.5, 2.5], [7.5, 2.5], [7.5, 7.5], [2.5, 7.5]])
    )


def test_simplify_closed_polygon_keeps_input_when_it_would_degenerate() -> None:
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)

    simplified = simplify_closed_polygon(
        points, simplification=PolygonSimplification(epsilon=10.0)
    )

    assert simplified == pytest.approx(points)


def _bright_rect_image() -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    image = np.full((80, 80, 3), 40, dtype=np.uint8)
    truth = np.zeros((80, 80), dtype=bool)
    truth[20:60, 30:50] = True
    image[truth] = 230
    return image, truth


def test_merge_mask_fragments_fuses_split_pieces_into_one() -> None:
    mask = np.zeros((60, 40), dtype=bool)
    mask[10:25, 10:30] = True
    mask[28:50, 10:30] = True  # 3px gap, like a mask split across implant threads

    merged = merge_mask_fragments(mask, radius=3)

    assert scipy.ndimage.label(merged)[1] == 1
    assert merged[26, 20]


def test_merge_mask_fragments_zero_radius_is_noop() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:5, 2:5] = True
    mask[10:15, 10:15] = True

    assert np.array_equal(merge_mask_fragments(mask, radius=0), mask)


@pytest.mark.parametrize("method", ["chan_vese", "geodesic"])
def test_refine_mask_boundary_snaps_to_bright_object(method: str) -> None:
    image, truth = _bright_rect_image()
    gray = image[..., 0].astype(np.float32) / 255
    init = np.zeros_like(truth)
    init[23:57, 33:47] = True  # 3px short on every side

    refined = refine_mask_boundary(
        init, gray, method=method, iterations=30, band=8
    )

    iou_before = (init & truth).sum() / (init | truth).sum()
    iou_after = (refined & truth).sum() / (refined | truth).sum()
    assert iou_after > iou_before
    assert iou_after > 0.9


def test_refine_mask_boundary_stays_within_band() -> None:
    gray = np.zeros((60, 60), dtype=np.float32)
    gray[:, 30:] = 1.0  # a large bright neighbor the mask must not leak into
    init = np.zeros((60, 60), dtype=bool)
    init[20:40, 32:38] = True

    refined = refine_mask_boundary(
        init, gray, method="chan_vese", iterations=50, band=4
    )

    outer = scipy.ndimage.binary_dilation(init, iterations=5)
    assert not (refined & ~outer).any()


def test_postprocess_mask_returns_padded_mask_and_shifted_origin() -> None:
    simplification = PolygonSimplification(merge_radius=3, refine_method="none")
    mask = np.ones((10, 10), dtype=bool)

    processed, origin = postprocess_mask(
        mask,
        origin_xy=np.array([50.0, 60.0]),
        image=None,
        simplification=simplification,
    )

    margin = simplification.mask_margin
    assert processed.shape == (10 + 2 * margin, 10 + 2 * margin)
    assert origin == pytest.approx([50.0 - margin, 60.0 - margin])
    assert processed[margin : margin + 10, margin : margin + 10].all()


def test_resimplify_polygon_merges_pinched_polygon_into_one_blob() -> None:
    # Two lobes joined by a 2px neck, like an implant traced as separate pieces.
    points = np.array(
        [
            [10, 10], [30, 10], [30, 28], [21, 28], [21, 32], [30, 32],
            [30, 50], [10, 50], [10, 32], [19, 32], [19, 28], [10, 28],
        ],
        dtype=np.float64,
    )

    # Densify like an AI-traced outline (about one vertex per pixel).
    closed = np.r_[points, points[:1]]
    points = np.concatenate(
        [
            np.linspace(start, end, int(np.linalg.norm(end - start)) + 1)[:-1]
            for start, end in zip(closed[:-1], closed[1:], strict=True)
        ]
    )

    simplified = resimplify_polygon(
        points,
        image=None,
        simplification=PolygonSimplification(merge_radius=5, refine_method="none"),
    )

    mask = np.zeros((60, 40), dtype=bool)
    rr, cc = skimage.draw.polygon(simplified[:, 1], simplified[:, 0], mask.shape)
    mask[rr, cc] = True
    # The neck is filled out to (nearly) the full width.
    assert mask[30, 12:28].all()
    assert simplified[:, 1].min() == pytest.approx(10, abs=1.5)
    assert simplified[:, 1].max() == pytest.approx(50, abs=1.5)


def test_resimplify_polygon_never_adds_vertices() -> None:
    points = np.array([[10, 10], [40, 10], [40, 30], [10, 30]], dtype=np.float64)

    simplified = resimplify_polygon(
        points, image=None, simplification=PolygonSimplification()
    )

    assert simplified == pytest.approx(points)


def test_snap_vertices_to_edges_moves_vertices_onto_boundary() -> None:
    image, _truth = _bright_rect_image()  # bright rect x in [30, 50), y in [20, 60)
    # Outline 2px inside the left/right edges; the edge midpoints (indices 1, 3)
    # have horizontal normals, while corner normals are diagonal.
    points = np.array(
        [[32, 30], [32, 40], [32, 50], [47.5, 50], [47.5, 40], [47.5, 30]],
        dtype=np.float64,
    )

    snapped = snap_vertices_to_edges(points, image, radius=3.0)

    # The step edges lie between pixels 29|30 and 49|50.
    assert snapped[1] == pytest.approx([29.5, 40], abs=0.5)
    assert snapped[4] == pytest.approx([49.5, 40], abs=0.5)


def test_snap_vertices_to_edges_zero_radius_is_noop() -> None:
    image, _truth = _bright_rect_image()
    points = np.array([[32, 30], [47, 30], [47, 50], [32, 50]], dtype=np.float64)

    assert snap_vertices_to_edges(points, image, radius=0.0) == pytest.approx(points)
