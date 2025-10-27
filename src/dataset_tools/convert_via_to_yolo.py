#!/usr/bin/env python3
"""Convert VIA annotations to YOLO format.

This script supports both VIA 2.x and 3.x JSON exports.  In some exports
(`metadata`/`file` structure), the image width and height are not always
stored.  Previously this resulted in division by ``None`` during the YOLO
normalisation step.  We now gracefully recover the missing dimensions by
opening the corresponding image file.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description='Convert VIA annotations to YOLO labels')
    p.add_argument('--via_json', type=str, required=True, help='Path to VIA export JSON file')
    p.add_argument('--images_dir', type=str, required=True, help='Directory containing the source images')
    p.add_argument('--labels_dir', type=str, required=True, help='Destination directory for YOLO labels')
    p.add_argument('--class_map', nargs='+', required=True, help='Ordered list of class names to map to IDs')
    return p.parse_args()


def resolve_image(images_dir: Path, filename: str) -> Optional[Path]:
    """Return the best-effort path to an image given its filename."""

    candidates = [images_dir / filename, images_dir / Path(filename).name]
    for path in candidates:
        if path.exists():
            return path
    return None


def image_size(images_dir: Path, filename: str) -> Optional[Tuple[int, int]]:
    """Get the width and height of an image, if possible."""

    path = resolve_image(images_dir, filename)
    if not path:
        return None
    try:
        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        return None


def yolo_line(xc: float, yc: float, w: float, h: float, iw: int, ih: int, cls: int) -> str:
    return f"{cls} {xc/iw:.6f} {yc/ih:.6f} {w/iw:.6f} {h/ih:.6f}\n"


def write_label_file(labels_dir: Path, image_name: str, lines: Iterable[str]) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / (Path(image_name).stem + '.txt')).write_text(''.join(lines), encoding='utf-8')


def main():
    args = parse_args()
    images = Path(args.images_dir)
    labels = Path(args.labels_dir)
    labels.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.via_json).read_text(encoding='utf-8'))
    class_to_id = {c: i for i, c in enumerate(args.class_map)}

    if isinstance(data, dict) and 'metadata' in data and 'file' in data:
        files = data['file']
        meta = data['metadata']
        for _, m in meta.items():
            fid = str(m['fid'])
            file_info = files[fid]
            fname = file_info['fname']
            iw = file_info.get('width')
            ih = file_info.get('height')
            if iw is None or ih is None:
                size = image_size(images, fname)
                if not size:
                    continue
                iw, ih = size
            lines = []
            for reg in m.get('regions', []):
                if reg.get('type') != 'rect':
                    continue
                x, y, w, h = reg['x'], reg['y'], reg['width'], reg['height']
                xc, yc = x + w / 2.0, y + h / 2.0
                tags = reg.get('tags')
                label = tags[0] if tags else reg.get('title', '')
                if label not in class_to_id:
                    continue
                lines.append(yolo_line(xc, yc, w, h, iw, ih, class_to_id[label]))
            if lines:
                write_label_file(labels, fname, lines)
    else:
        for fname, item in data.items():
            iw = item.get('width')
            ih = item.get('height')
            if iw is None or ih is None:
                size = image_size(images, fname)
                if not size:
                    continue
                iw, ih = size
            lines = []
            for r in item.get('regions', []):
                s = r.get('shape_attributes', {})
                if s.get('name') != 'rect':
                    continue
                x, y, w, h = s['x'], s['y'], s['width'], s['height']
                xc, yc = x + w / 2.0, y + h / 2.0
                label = r.get('region_attributes', {}).get('class', '')
                if label not in class_to_id:
                    continue
                lines.append(yolo_line(xc, yc, w, h, iw, ih, class_to_id[label]))
            if lines:
                write_label_file(labels, fname, lines)
    print('Done. Labels at:', labels)

if __name__ == '__main__':
    main()
