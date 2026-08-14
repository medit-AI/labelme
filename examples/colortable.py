import numpy as np
from typing import Dict 
universal_to_fdi = np.array([
    0,  # Index 0
    18, 17, 16, 15, 14, 13, 12, 11,  # 1~8 (우상)
    21, 22, 23, 24, 25, 26, 27, 28,  # 9~16 (좌상)
    38, 37, 36, 35, 34, 33, 32, 31,  # 17~24 (좌하)
    41, 42, 43, 44, 45, 46, 47, 48   # 25~32 (우하)
])

universal_to_permanent = {
    "A": 4,
    "B": 5,
    "C": 6,
    "D": 7,
    "E": 8,
    
    "F": 9,
    "G": 10,
    "H": 11,
    "I": 12,
    "J": 13,
    
    "K": 20,
    "L": 21,
    "M": 22,
    "N": 23,
    "O": 24,
    
    "P": 25,
    "Q": 26,
    "R": 27,
    "S": 28,
    "T": 29,
}


    
def get_teeth_color_table(normalize=True, dark_mode=True, DARK_1ST=.25):
    """
    https://github.com/tensorflow/models/blob/master/research/deeplab/utils/get_dataset_colormap.py
    """
    
    color_table = np.asarray([
        [255, 255, 255],  # bg
        [6, 230, 230],  # 1
        [80, 50, 50],  # 2
        [4, 200, 3],  # 3
        [30, 20, 240],  # 4
        [240, 10, 7],  # 5
        [224, 5, 255],  # 6
        [235, 255, 7],  # 7
        [150, 5, 61],  # 8
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
    ])

    if dark_mode:
        norm_color_table = color_table / 255.0
        color_table2 = np.concatenate([norm_color_table[:10] for _ in range(5)], axis=0)

        norm_fdi0_color = norm_color_table[1:9]
        color_table2[np.arange(1, 9)] = norm_fdi0_color

        color_table2[np.arange(11, 19)] = norm_fdi0_color
        color_table2[np.arange(21, 29)] = (norm_fdi0_color - DARK_1ST).clip(0, 1)
        color_table2[np.arange(31, 39)] = (norm_fdi0_color - DARK_1ST).clip(0, 1)
        color_table2[np.arange(41, 49)] = norm_fdi0_color

        color_table = (color_table2 * 255).astype(np.uint8)
    if normalize:
        color_table = color_table / 255.
    else:
        color_table = color_table.astype(np.uint8)

    return color_table

colors = get_teeth_color_table(normalize=False, dark_mode=True, DARK_1ST=.25)


def get_universial_color_table(primary_brighter=0.3) -> Dict[str, np.ndarray]:
    """
    1~32 (영구치) 및 A~T (유치) 치아 번호에 대응하는 RGB 컬러 딕셔너리를 반환합니다.
    """
    # 1~32 영구치 색상 계산 , fdi color table
    color_table = get_teeth_color_table(normalize=False, dark_mode=True, DARK_1ST=.25)
    
    universal_color_table: Dict[str, np.ndarray] = {}

    # uni
    # 1. 영구치 (1 ~ 32) 매핑
    for num in range(1, 33):
        fdi = universal_to_fdi[num]
        universal_color_table[str(num)] = color_table[fdi].copy()

    # 2. 유치 (A ~ T) 매핑 및 밝기 보정
    for primary_code, perm_idx in universal_to_permanent.items():
        perm_fdi = universal_to_fdi[perm_idx]
        base_color = color_table[perm_fdi].astype(np.float32)
        
        if primary_brighter != 0:
            # 밝기 증가 (255 범위 내 절삭)
            brightened_color = base_color + (primary_brighter * 255.0)
            primary_color = np.clip(brightened_color, 0, 255).astype(np.uint8)
        else:
            primary_color = base_color.astype(np.uint8)
            
        universal_color_table[primary_code] = primary_color

    return universal_color_table


res = get_universial_color_table()
print(res)

import yaml
with open('universal_color_table.yaml', 'w') as f:
    yaml.dump({k: v.tolist() for k, v in res.items()}, f, default_flow_style =None)
    
import cv2

# Column layout mirrors the "Universal numbering system table" reference:
# 16 columns per row, split into a left ("upper/lower right") and right
# ("upper/lower left") half of 8 columns each.
_NUM_COLS = 16

_PERMANENT_ROWS = [
    [str(n) for n in range(1, 17)],
    [str(n) for n in range(32, 16, -1)],
]
_PRIMARY_ROWS = [
    [None, None, "A", "B", "C", "D", "E", None, "F", "G", "H", "I", "J", None, None, None],
    [None, None, "T", "S", "R", "Q", "P", None, "O", "N", "M", "L", "K", None, None, None],
]
_ALTERNATE_ROWS = [
    [None, None, "1d", "2d", "3d", "4d", "5d", None, "6d", "7d", "8d", "9d", "10d", None, None, None],
    [None, None, "20d", "19d", "18d", "17d", "16d", None, "15d", "14d", "13d", "12d", "11d", None, None, None],
]

# (section title, cell labels, labels used to look up the color).
# The alternate system numbers the same teeth as the primary letters, so its
# cells are colored by the letter sitting at the same row/column.
_SECTIONS = [
    ("Permanent dentition", _PERMANENT_ROWS, _PERMANENT_ROWS),
    ("Primary dentition", _PRIMARY_ROWS, _PRIMARY_ROWS),
    ("Alternate system for primary dentition", _ALTERNATE_ROWS, _PRIMARY_ROWS),
]


def _luminance(rgb) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _draw_centered_text(image, x0, x1, y_baseline, text, font_scale, thickness, color=(0, 0, 0)):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = x0 + ((x1 - x0) - tw) // 2
    cv2.putText(
        image, text, (tx, y_baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
        thickness, cv2.LINE_AA,
    )


def _draw_cell(image, x0, y0, cell_size, label, rgb, font_scale):
    x1, y1 = x0 + cell_size, y0 + cell_size
    if rgb is not None:
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(image, (x0, y0), (x1, y1), bgr, thickness=-1)
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), thickness=1)
    if label is None:
        return
    text_color = (0, 0, 0) if rgb is None or _luminance(rgb) > 140 else (255, 255, 255)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.putText(
        image, label,
        (x0 + (cell_size - tw) // 2, y0 + (cell_size + th) // 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA,
    )


def create_color_table_image(
    color_table: Dict[str, np.ndarray],
    cell_size: int = 60,
    title_h: int = 50,
    section_title_h: int = 36,
    header_h: int = 30,
    footer_h: int = 30,
) -> np.ndarray:
    """Render the Universal Numbering System table with cells filled from color_table."""
    half = _NUM_COLS // 2
    width = _NUM_COLS * cell_size
    section_h = section_title_h + header_h + 2 * cell_size + footer_h
    height = title_h + len(_SECTIONS) * section_h

    image = np.full((height, width, 3), 255, dtype=np.uint8)

    _draw_centered_text(
        image, 0, width, title_h - 16, "Universal numbering system table",
        font_scale=0.9, thickness=2,
    )

    y = title_h
    for section_title, rows, color_rows in _SECTIONS:
        _draw_centered_text(
            image, 0, width, y + section_title_h - 10, section_title,
            font_scale=0.7, thickness=2,
        )
        y += section_title_h

        _draw_centered_text(image, 0, half * cell_size, y + header_h - 10, "upper right", 0.6, 1)
        _draw_centered_text(image, half * cell_size, width, y + header_h - 10, "upper left", 0.6, 1)
        y += header_h

        for row, color_row in zip(rows, color_rows):
            for col, label in enumerate(row):
                color_label = color_row[col]
                rgb = color_table.get(color_label) if color_label is not None else None
                _draw_cell(image, col * cell_size, y, cell_size, label, rgb, font_scale=0.55)
            y += cell_size

        _draw_centered_text(image, 0, half * cell_size, y + footer_h - 10, "lower right", 0.6, 1)
        _draw_centered_text(image, half * cell_size, width, y + footer_h - 10, "lower left", 0.6, 1)
        y += footer_h

    return image


color_table_image = create_color_table_image(color_table=res)

cv2.imwrite('color_table_image.png', color_table_image)