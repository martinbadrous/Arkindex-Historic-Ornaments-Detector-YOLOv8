#!/usr/bin/env bash
python src/dataset_tools/convert_via_to_yolo.py --via_json /data/ornaments/project.json --images_dir /data/ornaments/images --labels_dir /data/ornaments/labels --class_map lettrine illustration bandeau vignette
python src/dataset_tools/split_dataset.py --images_dir /data/ornaments/images --labels_dir /data/ornaments/labels --out_root /data/ornaments_yolo --val 0.2 --test 0.0
python src/train.py --data data/ornaments.yaml --model yolov8s.pt --imgsz 1024 --epochs 100 --batch 8
python src/arkindex_worker/worker.py --config src/arkindex_worker/config_arkindex.yaml --limit 10
