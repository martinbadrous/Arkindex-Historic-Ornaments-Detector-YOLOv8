#!/usr/bin/env python3
import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description='Train YOLOv8 for historic ornaments detection')
    p.add_argument('--data', type=str, default='data/ornaments.yaml')
    p.add_argument('--model', type=str, default='yolov8s.pt')
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--hyp', type=str, default=None)
    p.add_argument('--project', type=str, default='runs/ornaments')
    p.add_argument('--name', type=str, default='exp')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        cache=True,
        amp=True,
        deterministic=True,
        patience=args.patience,
        seed=args.seed,
        cfg=args.hyp
    )
    print(results)

if __name__ == '__main__':
    main()
