#!/usr/bin/env python3
import argparse, random, shutil
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description='Split YOLO dataset')
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--labels_dir', type=str, required=True)
    p.add_argument('--out_root', type=str, required=True)
    p.add_argument('--val', type=float, default=0.2)
    p.add_argument('--test', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    imgs = sorted([p for p in Path(args.images_dir).glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'}])
    assert imgs, 'No images found'
    n = len(imgs); idx = list(range(n)); random.shuffle(idx)
    n_test = int(n * args.test); n_val = int(n * args.val)
    test_set = set(idx[:n_test]); val_set = set(idx[n_test:n_test+n_val]); train_set = set(idx[n_test+n_val:])
    def outdirs(split):
        base = Path(args.out_root)/split
        (base/'images').mkdir(parents=True, exist_ok=True)
        (base/'labels').mkdir(parents=True, exist_ok=True)
        return base
    dtr, dva, dte = outdirs('train'), outdirs('val'), (outdirs('test') if args.test>0 else None)
    for i, img in enumerate(imgs):
        lab = Path(args.labels_dir)/(img.stem + '.txt')
        dst = dtr if i in train_set else (dva if i in val_set else (dte or dva))
        shutil.copy2(img, dst/'images'/img.name)
        if lab.exists(): shutil.copy2(lab, dst/'labels'/lab.name)
    print('Split done ->', args.out_root)

if __name__ == '__main__':
    main()
