#!/usr/bin/env python3
import argparse, yaml, os
from pathlib import Path
from ultralytics import YOLO
from .client import ArkindexClient

CLASSES_DEFAULT = ['lettrine','illustration','bandeau','vignette']

def parse_args():
    p = argparse.ArgumentParser(description='Arkindex Worker for Historic Ornaments (YOLOv8)')
    p.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'config_arkindex.yaml'))
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--imgsz', type=int, default=1024)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    api_url = cfg['api_url']; api_token = cfg['api_token']
    workspace_id = cfg['workspace_id']
    download_dir = cfg.get('download_dir', 'downloads')
    weights = cfg['model_weights']
    class_names = cfg.get('classes', CLASSES_DEFAULT)
    endpoints = cfg.get('endpoints', None)

    client = ArkindexClient(api_url, api_token, endpoints=endpoints)
    model = YOLO(weights)
    items = client.list_items(workspace_id, limit=args.limit)
    if not items:
        print('No pending items found.'); return
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    for item in items:
        try:
            print(f"[Item {item.get('id')}] Downloading...")
            img_path = client.download_image(item, download_dir)
            print(f"[Item {item.get('id')}] Inference...")
            res_list = model.predict(source=img_path, imgsz=args.imgsz, conf=args.conf, save=False, verbose=False)
            detections = []
            for r in res_list:
                if not hasattr(r, 'boxes') or r.boxes is None: continue
                for b in r.boxes:
                    x1,y1,x2,y2 = [float(v) for v in b.xyxy[0]]
                    w = x2 - x1; h = y2 - y1
                    cls_id = int(b.cls) if hasattr(b, 'cls') else 0
                    conf = float(b.conf) if hasattr(b, 'conf') else 0.0
                    name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
                    detections.append({'x':x1,'y':y1,'width':w,'height':h,'class':name,'confidence':conf})
            print(f"[Item {item.get('id')}] Posting {len(detections)} detections...")
            client.post_annotations(item.get('id'), detections)
            client.mark_done(item.get('id'), status='processed')
            print(f"[Item {item.get('id')}] ✅ Done.")
        except Exception as e:
            print(f"[Item {item.get('id')}] ❌ Error: {e}")

if __name__ == '__main__':
    main()
