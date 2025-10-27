# 📚 Arkindex Historic Ornaments Detector — YOLOv8

**Automatic detection of lettrines, illustrations, bandeaux, and vignettes in digitized ancient books**,  
integrated with **Teklia’s Arkindex platform** for large-scale processing and annotation management.

Developed by **[Martin Badrous](https://github.com/martinbadrous)** as part of his work at **Polytech Tours**.

---

## 🚀 Overview

This project provides a full deep learning pipeline for detecting **ornamental and illustrative elements** in scanned historical documents.

It combines:
- **YOLOv8** for real-time detection of ornaments (`lettrine`, `illustration`, `bandeau`, `vignette`)
- **Arkindex API integration** for automatic page retrieval and annotation upload
- Custom preprocessing and dataset conversion tools (VIA → YOLO)

---

## 🧱 Repository Structure

```bash
Arkindex-Historic-Ornaments-Detector-YOLOv8/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── ornaments.yaml              # YOLO dataset config
│
├── src/
│   ├── train.py                    # Train YOLOv8
│   ├── eval.py                     # Evaluate trained models
│   ├── infer.py                    # Run inference on images or folders
│   ├── dataset_tools/
│   │   ├── convert_via_to_yolo.py  # VIA 2.x JSON → YOLO labels
│   │   ├── split_dataset.py        # Train/Val/Test split
│   │   └── visualize_annotations.py# Visualization for verification
│   └── arkindex_worker/
│       ├── worker.py               # Main Arkindex processing script
│       ├── client.py               # Minimal Arkindex REST client
│       └── config_arkindex.yaml    # Configuration for API + model
│
├── configs/
│   ├── hyp_augment.yaml            # Training hyperparameters
│   └── config_arkindex.yaml        # Duplicate for convenience
│
└── scripts/
    └── demo_commands.sh            # Example usage
```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/martinbadrous/Arkindex-Historic-Ornaments-Detector-YOLOv8.git
cd Arkindex-Historic-Ornaments-Detector-YOLOv8

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧩 Dataset Preparation

### Supported annotation format
- [**VIA 2.x**](https://www.robots.ox.ac.uk/~vgg/software/via/) JSON format (Visual Image Annotator)
- Converts to YOLO bounding boxes using `convert_via_to_yolo.py`

### Example command:
```bash
python src/dataset_tools/convert_via_to_yolo.py   --via_json /path/to/project.json   --images_dir /path/to/images   --labels_dir /path/to/labels   --class_map lettrine illustration bandeau vignette
```

### Split dataset
```bash
python src/dataset_tools/split_dataset.py   --images_dir /path/to/images   --labels_dir /path/to/labels   --out_root /path/to/dataset_yolo   --val 0.2
```

### Edit dataset config
Update `data/ornaments.yaml` to match your dataset paths.

---

## 🧠 Training the Detector

Train YOLOv8 on your historical ornament dataset:

```bash
python src/train.py   --data data/ornaments.yaml   --model yolov8s.pt   --imgsz 1024 --epochs 100 --batch 8   --project runs/ornaments --name v8s_baseline
```

🧾 Training logs and weights will be saved to `runs/ornaments/v8s_baseline/`.

---

## 🔍 Inference

Predict ornaments on new images or folders:

```bash
python src/infer.py   --weights runs/ornaments/v8s_baseline/weights/best.pt   --source /path/to/test_images   --imgsz 1024 --conf 0.25 --save_txt --save_conf
```

Predictions (bounding boxes + labels) are saved under `runs/predict/`.

---

## 🧬 Arkindex Integration (Teklia)

The `arkindex_worker` folder allows the detector to **connect to an Arkindex instance**, automatically process pages, and upload annotations.

### Configure connection
Edit `src/arkindex_worker/config_arkindex.yaml`:

```yaml
api_url: "https://arkindex.teklia.com/api"
api_token: "YOUR_API_TOKEN"
workspace_id: "YOUR_WORKSPACE_ID"
download_dir: "downloads"
model_weights: "runs/ornaments/v8s_baseline/weights/best.pt"
classes: ["lettrine","illustration","bandeau","vignette"]

endpoints:
  list_items: "/workspaces/{workspace_id}/items?status=pending&limit=10"
  download_image: "/items/{item_id}/content"
  post_annotations: "/items/{item_id}/annotations"
  mark_done: "/items/{item_id}/status"
```

### Run worker
```bash
python src/arkindex_worker/worker.py   --config src/arkindex_worker/config_arkindex.yaml   --limit 10
```

This script will:
1. Pull pending pages from Arkindex.  
2. Download each image locally.  
3. Run YOLOv8 detection.  
4. Upload detections as bounding box annotations.  
5. Mark items as processed.

---

## 🧰 Example Results

| Class         | Example |
|----------------|----------|
| **Lettrine**   | Large decorative initials |
| **Illustration** | Inline engravings or images |
| **Bandeau**    | Decorative horizontal bands |
| **Vignette**   | Small ornamental graphics |

---

## 🧪 Notes for Historical Documents

- Use high-resolution scans (≥1000px width).  
- Apply mild denoising or contrast enhancement for degraded pages.  
- Fine-tune augmentations via `configs/hyp_augment.yaml`.  
- For small elements like lettrines, use larger images and higher input size.

---

## 📈 Future Work

- Add instance segmentation for ornament contours  
- Extend worker for parallel processing in Arkindex  
- Integrate OCR to classify decorated initials by letter  

---

## 👨‍💻 Author

**Martin Badrous**  
*Computer Vision & Deep Learning Engineer*  
🎓 M.Sc. Computer Vision & Robotics — Université de Bourgogne  
📧 martin.badrous@gmail.com  
🔗 [GitHub Profile](https://github.com/martinbadrous)

---

## 🪪 License

MIT License © 2025 Martin Badrous  
Free to use for research and educational purposes.
