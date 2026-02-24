# 🔍 OCR Model — DBNet + CRNN Pipeline

An end-to-end **Optical Character Recognition (OCR)** pipeline built from scratch in **PyTorch**, combining **DBNet** for text detection and **CRNN** for text recognition. The project demonstrates text localization using differentiable binarization and word-level recognition using CNN-RNN hybrid architectures with CTC decoding.

---

## 📐 Architecture Overview

```
Input Image
    │
    ▼
┌──────────────────────────────────────┐
│          TEXT DETECTION (DBNet)       │
│                                      │
│  ResNet-18 Backbone                  │
│       │                              │
│  Feature Pyramid Network (FPN)       │
│       │                              │
│  DBNet Head (Probability + Threshold │
│       + Binary Maps)                 │
│       │                              │
│  Post-processing → Bounding Boxes    │
└──────────────────────────────────────┘
    │
    ▼  Cropped word regions
┌──────────────────────────────────────┐
│       TEXT RECOGNITION (CRNN)        │
│                                      │
│  CNN Feature Extractor               │
│       │                              │
│  Bidirectional LSTM Sequence Encoder │
│       │                              │
│  CTC Decoder → Recognized Text      │
└──────────────────────────────────────┘
    │
    ▼
Recognized Text Output
```

---

## 📁 Project Structure

```
OCR-model-dbnet-RCNN/
│
├── single word recognition.ipynb         # CRNN training on single image word regions
├── multi image recognition.ipynb         # CRNN training across multiple images
├── overfitting_single_image_dbnet.ipynb  # DBNet text detection (single image overfitting)
├── multiple image with checkbox.ipynb    # Full DBNet detection pipeline (multi-image)
│
├── single_image_data/
│   ├── img_single.png                    # Sample input image
│   └── img_single.json                   # Word-level polygon annotations
│
└── multiple_image_data/
    ├── *.png                             # Multiple input images (screenshots)
    └── multiple_image.json               # Annotations for all images
```

---

## 🧠 Key Components

### Text Detection — DBNet (Differentiable Binarization Network)

| Component | Description |
|---|---|
| **ResNet-18 Backbone** | Extracts multi-scale feature maps from input images |
| **FPN (Feature Pyramid Network)** | Fuses features across scales for robust detection |
| **DBNet Head** | Produces probability map, threshold map, and binary map |
| **Differentiable Binarization** | Learnable thresholding via `k * (P - T)` sigmoid |
| **Post-processing** | Contour extraction → minimum-area bounding boxes |

**Loss Functions:**
- **Dice Loss** — for binary/probability map supervision
- **OHEM Balanced BCE Loss** — hard example mining for threshold map
- **Combined DBNetLoss** — weighted sum of probability + binary + threshold losses

### Text Recognition — CRNN (CNN + RNN)

| Component | Description |
|---|---|
| **CNN Feature Extractor** | Convolutional layers to extract visual features from cropped word images |
| **Bidirectional LSTM** | Sequence modeling over feature columns |
| **CTC Decoder** | Connectionist Temporal Classification for alignment-free text decoding |
| **CharacterSet** | Character encoding/decoding utility (alphanumeric + special chars) |

---

## 📓 Notebooks

### 1. `overfitting_single_image_dbnet.ipynb`
> **Goal:** Train DBNet to detect text regions by overfitting on a single annotated image.

- Implements the full DBNet architecture (ResNet-18 + FPN + DBNet Head)
- Uses polygon shrinking (`pyclipper`) to generate ground-truth probability and threshold maps
- Trains with combined Dice + OHEM-BCE loss
- Includes bounding box extraction and IoU evaluation
- **20 code cells** — most comprehensive detection notebook

### 2. `multiple image with checkbox.ipynb`
> **Goal:** Extend DBNet detection to multiple images with a full training and prediction pipeline.

- Same DBNet architecture as the overfitting notebook
- Custom `DBNetDataset` with `DataLoader` for multi-image training
- Prediction and visualization utilities (`predict_on_image`, `visualize_prediction`)
- **16 code cells**

### 3. `single word recognition.ipynb`
> **Goal:** Train a CRNN model to recognize individual words from cropped text regions.

- Implements `CRNN` with CNN + BiLSTM + linear projection
- Uses CTC loss for training without explicit character alignment
- Polygon-based word cropping with perspective transform (`crop_and_warp`)
- **13 code cells**

### 4. `multi image recognition.ipynb`
> **Goal:** Scale word recognition across multiple images using a dataset pipeline.

- Full `WordRecognitionDataset` with `DataLoader` and custom `collate_fn`
- `CTCDecoder` class for batch decoding predictions
- End-to-end inference via `recognize_word` function
- **15 code cells**

---

## ⚙️ Tech Stack & Dependencies

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework (model, training, loss) |
| `torchvision` | (Implicit via PyTorch ecosystem) |
| `opencv-python` (`cv2`) | Image loading, preprocessing, contour extraction |
| `numpy` | Numerical operations |
| `matplotlib` | Visualization of predictions, feature maps, and loss curves |
| `shapely` | Polygon geometry operations |
| `pyclipper` | Polygon shrinking for DBNet ground-truth generation |

### Installation

```bash
pip install torch torchvision opencv-python numpy matplotlib shapely pyclipper
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd OCR-model-dbnet-RCNN
```

### 2. Install Dependencies

```bash
pip install torch torchvision opencv-python numpy matplotlib shapely pyclipper
```

### 3. Run the Notebooks

Open any notebook in **Jupyter Notebook** or **VS Code**:

```bash
jupyter notebook
```

**Recommended order:**

1. **`overfitting_single_image_dbnet.ipynb`** — Understand DBNet text detection
2. **`multiple image with checkbox.ipynb`** — Scale detection to multiple images
3. **`single word recognition.ipynb`** — Learn CRNN word recognition
4. **`multi image recognition.ipynb`** — Full multi-image recognition pipeline

---

## 📊 Data Format

Annotations follow a **JSON format** with polygon-based word bounding boxes:

```json
{
  "image_path": "path/to/image.png",
  "words": [
    {
      "text": "example",
      "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    }
  ]
}
```

- **Polygons** define four-point quadrilateral regions around each word
- **Text** provides the ground-truth transcription for recognition training

---

## 🔬 How It Works

### Detection Phase (DBNet)
1. Input image is resized and passed through **ResNet-18** to extract multi-scale features
2. **FPN** fuses features from different levels into a unified representation
3. **DBNet Head** predicts three maps:
   - **Probability Map** — likelihood of each pixel being text
   - **Threshold Map** — adaptive threshold per pixel
   - **Binary Map** — final segmentation via differentiable binarization
4. **Post-processing** extracts contours from the binary map and fits minimum-area bounding boxes

### Recognition Phase (CRNN)
1. Detected word regions are **cropped and perspective-corrected** using polygon coordinates
2. Cropped images are resized to a fixed height (32px) and passed through **CNN layers**
3. Feature columns are fed into a **Bidirectional LSTM** for sequence modeling
4. **CTC Decoder** converts the output sequence into readable text, handling alignment and deduplication

---

## 📝 License

This project is for educational and research purposes.

