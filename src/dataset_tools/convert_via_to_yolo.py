#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--via_json', type=str, required=True)
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--labels_dir', type=str, required=True)
    p.add_argument('--class_map', nargs='+', required=True)
    return p.parse_args()

def yolo_line(xc,yc,w,h,iw,ih,cls):
    return f"{cls} {xc/iw:.6f} {yc/ih:.6f} {w/iw:.6f} {h/ih:.6f}\n"

def main():
    args = parse_args()
    labels = Path(args.labels_dir); labels.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.via_json).read_text(encoding='utf-8'))
    class_to_id = {c:i for i,c in enumerate(args.class_map)}
    if isinstance(data, dict) and 'metadata' in data and 'file' in data:
        files = data['file']; meta = data['metadata']
        for _, m in meta.items():
            fid = str(m['fid']); fname = files[fid]['fname']
            iw = files[fid].get('width'); ih = files[fid].get('height')
            lines = []
            for reg in m.get('regions', []):
                if reg.get('type') != 'rect': continue
                x,y,w,h = reg['x'],reg['y'],reg['width'],reg['height']
                xc,yc = x+w/2.0, y+h/2.0
                label = reg.get('tags', [''])[0] if reg.get('tags') else reg.get('title','')
                if label not in class_to_id: continue
                lines.append(yolo_line(xc,yc,w,h,iw,ih,class_to_id[label]))
            if lines:
                (labels/(Path(fname).stem + '.txt')).write_text(''.join(lines), encoding='utf-8')
    else:
        for fname, item in data.items():
            regions = item.get('regions', [])
            iw = item.get('width'); ih = item.get('height')
            if iw is None or ih is None:
                try:
                    import cv2
                    im = cv2.imread(str(Path(args.images_dir)/fname)); ih, iw = im.shape[:2]
                except Exception:
                    continue
            lines = []
            for r in regions:
                s = r.get('shape_attributes', {})
                if s.get('name') != 'rect': continue
                x,y,w,h = s['x'],s['y'],s['width'],s['height']
                xc,yc = x+w/2.0, y+h/2.0
                label = r.get('region_attributes', {}).get('class','')
                if label not in class_to_id: continue
                lines.append(yolo_line(xc,yc,w,h,iw,ih,class_to_id[label]))
            if lines:
                (labels/(Path(fname).stem + '.txt')).write_text(''.join(lines), encoding='utf-8')
    print('Done. Labels at:', labels)

if __name__ == '__main__':
    main()
