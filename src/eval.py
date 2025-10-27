#!/usr/bin/env python3
import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate YOLOv8 ornaments detector')
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--data', type=str, default='data/ornaments.yaml')
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--batch', type=int, default=8)
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, plots=True)
    print(metrics)

if __name__ == '__main__':
    main()
