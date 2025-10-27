#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2

def parse_args():
    p = argparse.ArgumentParser(description='Visualize YOLO annotations')
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--labels_dir', type=str, required=True)
    p.add_argument('--names', nargs='+', default=['lettrine','illustration','bandeau','vignette'])
    p.add_argument('--out_dir', type=str, default='viz')
    p.add_argument('--max', type=int, default=100)
    return p.parse_args()

def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in sorted(Path(args.images_dir).glob('*')):
        if img_path.suffix.lower() not in {'.jpg','.jpeg','.png','.tif','.tiff'}: continue
        lab_path = Path(args.labels_dir)/(img_path.stem + '.txt')
        if not lab_path.exists(): continue
        im = cv2.imread(str(img_path)); 
        if im is None: continue
        h,w = im.shape[:2]
        for line in lab_path.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            cls, xc, yc, bw, bh = map(float, line.strip().split())
            cls = int(cls)
            x = int((xc - bw/2) * w); y = int((yc - bh/2) * h)
            ww = int(bw * w); hh = int(bh * h)
            cv2.rectangle(im, (x,y), (x+ww, y+hh), (0,255,0), 2)
            label = args.names[cls] if 0 <= cls < len(args.names) else str(cls)
            cv2.putText(im, label, (x, max(0,y-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imwrite(str(out/img_path.name), im)
        count += 1
        if count >= args.max: break
    print('Saved', count, 'visualizations to', out)

if __name__ == '__main__':
    main()
