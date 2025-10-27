#!/usr/bin/env python3
import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description='Inference with YOLOv8 ornaments detector')
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--source', type=str, required=True)
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--device', type=str, default='')
    p.add_argument('--save_txt', action='store_true')
    p.add_argument('--save_conf', action='store_true')
    p.add_argument('--project', type=str, default='runs/predict')
    p.add_argument('--name', type=str, default='exp')
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name
    )
    print('Predictions saved.')

if __name__ == '__main__':
    main()
