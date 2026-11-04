#!/usr/bin/env python
import tqdm
import base64
import os
import argparse
import collections
import datetime
import json
import math
import random
import sys
import types
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import Final
from typing import List

import imgviz
import numpy as np

try:
    import pycocotools.mask  # type: ignore
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils  # noqa: E402  # examples/utils.py, vendored alongside this script


def _circle_to_polygon_segmentation(
    *, center: tuple[float, float], edge: tuple[float, float]
) -> list[float]:
    CHORD_TOLERANCE_PX: Final = 1.0
    MIN_VERTICES: Final = 12

    cx, cy = center
    ex, ey = edge
    radius = math.hypot(ex - cx, ey - cy)
    if radius == 0.0:
        raise ValueError("Degenerate circle: center and edge are the same point.")

    # Pick a vertex count so the chord-to-arc deviation stays within 1 pixel:
    # solving r * (1 - cos(pi / n)) <= tol for n yields n >= pi / acos(1 - tol/r).
    # When r <= tol the formula domain breaks, so clamp to the minimum.
    if radius <= CHORD_TOLERANCE_PX:
        n_vertices = MIN_VERTICES
    else:
        n_vertices = max(
            MIN_VERTICES,
            int(math.pi / math.acos(1.0 - CHORD_TOLERANCE_PX / radius)),
        )
    angles = (2.0 * math.pi / n_vertices) * np.arange(n_vertices)
    coords = np.empty((n_vertices, 2), dtype=float)
    coords[:, 0] = cx + radius * np.cos(angles)
    coords[:, 1] = cy + radius * np.sin(angles)
    return coords.flatten().tolist()

def deep_search_files(basepath, exts=[]) -> List[str]:
    """
    파일 딥 서치
    Parameters
    ----------
    basepath :
    exts :

    Returns
    -------

    """
    searched = []

    for root, dirs, files in os.walk(basepath):
        for ext in exts:
            file_pathes = [os.path.join(root, file) for file in files if isinstance(ext, str) and file.endswith(ext)]
            searched.extend(file_pathes)
    return searched



def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        print("Output directory already exists:", output_dir)
        sys.exit(1)
    output_dir.mkdir(parents=True)
    (output_dir / "JPEGImages").mkdir(parents=True)
    if not args.noviz:
        (output_dir / "Visualization").mkdir(parents=True)
    print("Creating dataset:", output_dir)

    now = datetime.datetime.now()

    data: dict[str, Any] = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[
            dict(
                url=None,
                id=0,
                name=None,
            )
        ],
        type="instances",
    )
    data["images"] = []  # license, url, file_name, height, width, date_captured, id
    data["categories"] = []  # supercategory, id, name
    data[
        "annotations"
    ] = []  # segmentation, area, iscrowd, image_id, bbox, category_id, id

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(
                supercategory=None,
                id=class_id,
                name=class_name,
            )
        )

    out_ann_file = output_dir / "annotations.json"
    label_files = sorted(Path(args.input_dir).glob("*.json"))
    for image_id, path in enumerate(label_files):
        print("Generating dataset from:", path)

        label_file = utils.load_label_file(str(path))

        base = path.stem
        out_img_file = output_dir / "JPEGImages" / f"{base}.jpg"

        img = utils.img_data_to_arr(label_file.image_data)
        if img.ndim == 3 and img.shape[2] == 4:
            img = imgviz.rgba2rgb(img)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=str(out_img_file.relative_to(output_dir)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            points: list[list[int | float]] = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = utils.shape_to_mask(img.shape[:2], points, shape_type)

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            points_coco: list[int | float]
            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points_coco = [x1, y1, x2, y1, x2, y2, x1, y2]
            elif shape_type == "circle":
                (cx, cy), (ex, ey) = points
                points_coco = _circle_to_polygon_segmentation(
                    center=(cx, cy), edge=(ex, ey)
                )
            else:
                points_coco = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points_coco)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        if not args.noviz:
            viz = img
            if masks:
                labels, captions, masks = zip(
                    *[
                        (class_name_to_id[cnm], cnm, msk)
                        for (cnm, gid), msk in masks.items()
                        if cnm in class_name_to_id
                    ]
                )
                viz = imgviz.instances2rgb(
                    image=img,
                    labels=labels,
                    masks=masks,
                    captions=captions,
                    font_size=15,
                    line_width=2,
                )
            out_viz_file = output_dir / "Visualization" / f"{base}.jpg"
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


_IMAGE_EXTS: Final = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _find_image_for_json(json_path: Path) -> Path | None:
    """Find the image saved next to json_path under the labelme convention:
    identical filename, common raster extension, case-insensitive (real
    datasets mix .png/.PNG etc). The json's own imagePath field is ignored on
    purpose -- it easily goes stale once a dataset is copied to another
    drive/machine (e.g. into a synced Dropbox folder).
    """
    for candidate in json_path.parent.iterdir():
        if candidate.stem == json_path.stem and candidate.suffix.lower() in _IMAGE_EXTS:
            return candidate
    return None


def _load_pair(json_path: Path, image_path: Path) -> utils.LabeledImage:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("imageData") is not None:
        image_data = base64.b64decode(data["imageData"])
    else:
        # Many datasets are saved without embedded imageData to save space,
        # so fall back to the image file actually matched on disk.
        image_data = image_path.read_bytes()
    shapes = [
        {
            "label": shape["label"],
            "points": shape["points"],
            "shape_type": shape.get("shape_type") or "polygon",
            "group_id": shape.get("group_id"),
            "flags": shape.get("flags") or {},
        }
        for shape in data["shapes"]
    ]
    return utils.LabeledImage(image_data=image_data, shapes=shapes)


def _find_labelme_pairs(
    input_dirs: str | Path | list[str | Path],
) -> list[tuple[Path, Path]]:
    """Deep-search input_dirs for labelme .json files that have an image of
    the identical filename saved in the same folder, and return (json_path,
    image_path) for every pair found.
    """
    if isinstance(input_dirs, (str, Path)):
        input_dirs = [input_dirs]

    pairs: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for input_dir in input_dirs:
        if not Path(input_dir).is_dir():
            print(f"WARNING: input dir does not exist (quote paths with spaces?): {input_dir!r}")
            continue
        json_paths = deep_search_files(basepath=str(input_dir), exts=[".json"])
        print(f"{input_dir}: found {len(json_paths)} json files")
        for json_path in sorted(Path(p) for p in json_paths):
            if json_path in seen:
                continue
            seen.add(json_path)

            image_path = _find_image_for_json(json_path)
            if image_path is None:
                print(f"Skipping {json_path}: no matching image found next to it")
                continue
            pairs.append((json_path, image_path))
    return pairs


_DEFAULT_SPLITS: Final = types.MappingProxyType({"train": 0.9, "valid": 0.1})


def _split_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Split total items by weights (any positive scale, need not sum to 1).

    Largest-remainder rounding, so the counts always add up to total exactly.
    """
    if not weights or any(w < 0 for w in weights.values()):
        raise ValueError(f"split weights must be non-negative: {dict(weights)}")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError(f"split weights must not all be zero: {dict(weights)}")
    exact = {name: total * w / weight_sum for name, w in weights.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    by_remainder = sorted(exact, key=lambda name: exact[name] - counts[name], reverse=True)
    for name in by_remainder[: total - sum(counts.values())]:
        counts[name] += 1
    return counts


def convert_labelme_dirs_to_coco(
    input_dirs: str | Path | list[str | Path],
    output_dir: str | Path,
    labels_file: str | Path | None = None,
    noviz: bool = False,
    color_map: str | None = None,
    splits: Mapping[str, float] | None = _DEFAULT_SPLITS,
    seed: int = 0,
) -> dict[str, Path]:
    """Deep-search input_dirs for labelme image/json pairs (image and .json
    saved together under the same filename) and convert every pair found
    into COCO datasets.

    labels_file is optional. Without it, the category list is collected
    straight from the labels already present in the discovered jsons'
    shapes -- labelme output has no separate text label list to convert,
    it's already all in the json.

    splits maps a split name to its weight, e.g. {"train": 0.8, "valid": 0.1,
    "test": 0.1}. The pairs are shuffled with seed and divided by those
    weights, and each split is saved as its own COCO dataset under
    output_dir/<split name>/ (JPEGImages, Visualization, annotations.json),
    all sharing the same categories. output_dir/splits.json records which
    source json went to which split. splits=None writes a single dataset
    directly into output_dir, as before.

    Returns the annotations.json path of each split (key "all" for None).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Creating dataset:", output_dir)

    pairs = _find_labelme_pairs(input_dirs=input_dirs)
    print(f"Found {len(pairs)} labelme image/json pairs")

    loaded = [
        (json_path, image_path, _load_pair(json_path, image_path))
        for json_path, image_path in pairs
    ]

    if labels_file is not None:
        class_name_to_id = {}
        for i, line in enumerate(open(labels_file).readlines()):
            class_id = i - 1  # starts with -1
            class_name = line.strip()
            if class_id == -1:
                assert class_name == "__ignore__"
                continue
            class_name_to_id[class_name] = class_id
    else:
        label_names = sorted(
            {shape["label"] for _, _, label_file in loaded for shape in label_file.shapes}
        )
        digit_class_max = 1000
        class_name_to_id = dict()
        for i, name in enumerate(label_names):
            if name.isdigit():
                class_name_to_id[name] = int(name)
            else:
                class_name_to_id[name] = digit_class_max + i
                
            
    color_map_table = (
        get_teeth_color_table(
            normalize=False,
            dark_mode=True,
            DARK_1ST=0.25,
            size=max(class_name_to_id.values(), default=0) + 1,
        )
        if color_map == "teeth"
        else None
    )

            # class_name_to_id = {name: i for i, name in enumerate(label_names)}
        # class_name_to_id = {name: i for i, name in enumerate(label_names)}

    if splits is None:
        out_ann_file = _write_coco_dataset(
            loaded,
            output_dir=output_dir,
            class_name_to_id=class_name_to_id,
            noviz=noviz,
            color_map_table=color_map_table,
        )
        return {"all": out_ann_file}

    # Shuffle a copy with a fixed seed so the same inputs always give the same
    # split, independent of the order the files were found in.
    shuffled = list(loaded)
    random.Random(seed).shuffle(shuffled)
    counts = _split_counts(total=len(shuffled), weights=splits)

    out_ann_files: dict[str, Path] = {}
    manifest: dict[str, list[str]] = {}
    start = 0
    for split_name, count in counts.items():
        split_items = shuffled[start : start + count]
        start += count
        print(f"Split {split_name!r}: {count} pairs")
        manifest[split_name] = [str(json_path) for json_path, _, _ in split_items]
        out_ann_files[split_name] = _write_coco_dataset(
            split_items,
            output_dir=output_dir / split_name,
            class_name_to_id=class_name_to_id,
            noviz=noviz,
            color_map_table=color_map_table,
        )

    manifest_file = output_dir / "splits.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(
            dict(seed=seed, weights=dict(splits), counts=counts, files=manifest),
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("Saved split manifest to:", manifest_file)
    return out_ann_files


def _write_coco_dataset(
    items: list[tuple[Path, Path, utils.LabeledImage]],
    output_dir: Path,
    class_name_to_id: dict[str, int],
    noviz: bool,
    color_map_table: np.ndarray | None,
) -> Path:
    """Convert (json_path, image_path, label_file) items into one COCO dataset
    under output_dir: JPEGImages/, Visualization/ and annotations.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "JPEGImages").mkdir(parents=True, exist_ok=True)
    if not noviz:
        (output_dir / "Visualization").mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now()
    data: dict[str, Any] = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None)],
        type="instances",
    )
    data["images"] = []
    data["categories"] = [
        dict(supercategory=None, id=class_id, name=class_name)
        for class_name, class_id in class_name_to_id.items()
    ]
    data["annotations"] = []

    out_ann_file = output_dir / "annotations.json"
    test_break = 10
    for image_id, (json_path, image_path, label_file) in tqdm.tqdm(
        enumerate(items), total=len(items)
    ):
        # if image_id >= test_break:
            # break
        print("Generating dataset from:", json_path)

        # Different input_dirs can contain same-named files in different
        # sub-folders, so the output name is disambiguated with image_id
        # instead of reusing the stem alone.
        base = f"{image_id:06d}_{image_path.stem}"
        out_img_file = output_dir / "JPEGImages" / f"{base}.jpg"

        img = utils.img_data_to_arr(label_file.image_data)
        if img.ndim == 3 and img.shape[2] == 4:
            img = imgviz.rgba2rgb(img)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=str(out_img_file.relative_to(output_dir)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            points: list[list[int | float]] = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = utils.shape_to_mask(img.shape[:2], points, shape_type)

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            points_coco: list[int | float]
            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points_coco = [x1, y1, x2, y1, x2, y2, x1, y2]
            elif shape_type == "circle":
                (cx, cy), (ex, ey) = points
                points_coco = _circle_to_polygon_segmentation(
                    center=(cx, cy), edge=(ex, ey)
                )
            else:
                points_coco = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points_coco)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        if not noviz:
            viz = img
            if masks:
                labels, captions, viz_masks = zip(
                    *[
                        (class_name_to_id[cnm], cnm, msk)
                        for (cnm, gid), msk in masks.items()
                        if cnm in class_name_to_id
                    ]
                )
                viz = imgviz.instances2rgb(
                    image=img,
                    labels=labels,
                    masks=viz_masks,
                    captions=captions,
                    font_size=15,
                    line_width=2,
                    colormap=color_map_table,
                )
            out_viz_file = output_dir / "Visualization" / f"{base}.jpg"
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)

    print("Saved annotations to:", out_ann_file)
    return out_ann_file

    
# Universal numbering (1..32, labelme default) -> FDI two-digit number.
_UNIVERSAL_TO_FDI: Final = np.array(
    [0]
    + [18, 17, 16, 15, 14, 13, 12, 11]  # 1~8   (upper right)
    + [21, 22, 23, 24, 25, 26, 27, 28]  # 9~16  (upper left)
    + [38, 37, 36, 35, 34, 33, 32, 31]  # 17~24 (lower left)
    + [41, 42, 43, 44, 45, 46, 47, 48]  # 25~32 (lower right)
)

_TOOTH_COLORS: Final = np.asarray(
    [
        [6, 230, 230],  # 1 (central incisor)
        [80, 50, 50],  # 2
        [4, 200, 3],  # 3
        [30, 20, 240],  # 4
        [240, 10, 7],  # 5
        [224, 5, 255],  # 6
        [235, 255, 7],  # 7
        [150, 5, 61],  # 8 (third molar)
    ],
    dtype=np.float64,
)
_BG_COLOR: Final = (255, 255, 255)
_OTHER_COLOR: Final = (128, 128, 128)  # non-tooth labels (e.g. 1000+)


def get_teeth_color_table(
    normalize=False, dark_mode=True, DARK_1ST=0.25, as_fdi=False, size=None
):
    """Color table indexed directly by label id.

    as_fdi=False: label id is universal numbering (1..32, labelme default).
    as_fdi=True : label id is FDI numbering (11..18, 21..28, 31..38, 41..48).
    Each tooth gets the color of its tooth type (FDI last digit 1..8); with
    dark_mode the left-side quadrants (2, 3) are darkened by DARK_1ST.
    Ids that are not teeth (0 = bg, > 32 universal / invalid FDI) get
    white / gray. size is the table length (>= 49 and > the largest label id).
    """
    size = max(size or 0, 49)
    table = np.tile(np.asarray(_OTHER_COLOR, dtype=np.float64) / 255.0, (size, 1))
    table[0] = np.asarray(_BG_COLOR) / 255.0
    base = _TOOTH_COLORS / 255.0

    for universal in range(1, 33):
        fdi = int(_UNIVERSAL_TO_FDI[universal])
        quadrant, tooth = divmod(fdi, 10)
        color = base[tooth - 1]
        if dark_mode and quadrant in (2, 3):
            color = (color - DARK_1ST).clip(0, 1)
        table[fdi if as_fdi else universal] = color

    if normalize:
        return table
    return (table * 255).astype(np.uint8)


def _parse_splits(items: list[str]) -> dict[str, float]:
    splits: dict[str, float] = {}
    for item in items:
        name, sep, weight = item.partition("=")
        if not sep or not name:
            raise argparse.ArgumentTypeError(f"expected NAME=WEIGHT, got {item!r}")
        splits[name] = float(weight)
    return splits


def cli() -> None:
    default_splits = [f"{k}={v}" for k, v in _DEFAULT_SPLITS.items()]
    parser = argparse.ArgumentParser(
        description="Convert labelme image/json pairs (deep-searched) to COCO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="input annotated directories (quote paths containing spaces)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="output dataset directory"
    )
    parser.add_argument("--labels", default=None, help="labels file (optional)")
    parser.add_argument("--noviz", action="store_true", help="no visualization")
    parser.add_argument(
        "--color-map",
        default='teeth',
        choices=["teeth"],
        help="visualization color table (default: imgviz default)",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        metavar="NAME=WEIGHT",
        default=default_splits,
        help="split names and weights",
    )
    parser.add_argument(
        "--no-split", action="store_true", help="write a single dataset (no split)"
    )
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed for split")
    args = parser.parse_args()

    convert_labelme_dirs_to_coco(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        labels_file=args.labels,
        noviz=args.noviz,
        color_map=args.color_map,
        splits=None if args.no_split else _parse_splits(args.split),
        seed=args.seed,
    )


if __name__ == "__main__":
    cli()
