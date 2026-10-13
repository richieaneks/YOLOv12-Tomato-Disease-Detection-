# Performance Analysis of YOLOv12 for Multi-Class Tomato Disease Detection

[![Kaggle Dataset](https://shields.io)](https://www.kaggle.com/datasets/farukalam/tomato-leaf-diseasesdetection-computer-vision)
[![Model-YOLOv12](https://shields.io)](https://github.com)

This repository contains the implementation, technical documentation, and inference pipeline evaluated for our **MCE 531 (Machine Vision)** group engineering design project. The study benchmarks the modern, attention-centric **Standard YOLOv12** deep learning architecture for multi-class plant disease classification.

---

## 👥 Group 8 Team Members & Contributions

| S/N | Matric Number | Name | Primary Project Role |
| :--- | :--- | :--- | :--- |
| 1 | 2021/1/80140ET | **Temple Jenom Gideon** | Project Lead & Pipeline Orchestrator |
| 2 | 2021/1/82262ET | **Ismail Wando Yakubu** | Model Training & Hyperparameter Tuning |
| 3 | 2021/1/81906ET | **Harrison Afolabi Raiyetunbi** | Data Engineering & Augmentation Pipeline |
| 4 | 2021/1/79519ET | **Sunday Gabriel Abayomi** | Model Training & Hyperparameter Tuning |
| 5 | 2021/1/83070ET | **Abdullahi Kamilud-Deen M.** | Performance Evaluation & Metrics Analysis |
| 6 | 2022/2/85570ET | **Richard Chidubem Aniekwe** | Performance Evaluation, Technical Documentation & Compilation |
| 7 | 2021/1/79838ET | **Christian Owoicho-Oche Ameh** | Performance Evaluation & Metrics Analysis |

---

## 📝 Abstract
This study evaluates the performance of the **Standard YOLOv12** architecture for automated detection of multi-class tomato leaf diseases. The **Kaggle Tomato Leaf Diseases Detection** dataset, comprising 645 training images, 61 validation images, and 31 test images across seven target classes, was used to train and benchmark the model. Transfer learning from COCO pretrained weights was applied, and three advanced augmentation techniques, namely Mosaic, MixUp, and CutMix, were employed to improve generalisation in dense agricultural environments. A systematic hyperparameter study across five learning rates (0.1, 0.01, 0.001, 0.0001, and 0.00001) was conducted to determine optimal convergence behaviour. The learning rate of `0.00001` produced the best-documented validation results, achieving an overall mAP@0.5 of **0.762** and mAP@0.5:0.95 of **0.502**. The *Healthy* class reached near-perfect detection at mAP@0.5 of **0.995**, while *Early Blight* and *Late Blight* also demonstrated strong performance. These findings confirm that Standard YOLOv12 is a highly capable and scalable architecture suited for real-time embedded agrotech deployment.

---

## 📂 Repository Structure
```directory
├── results/               # Inference output bounding boxes and metrics
├── weights/               # Trained network model weight binaries (.pt)
│   └── best.pt            # Top performing model weights (LR = 0.00001)
├── src/
│   └── inference.py       # Standalone cross-platform model inference script
└── README.md              # Academic documentation and report summary
```

---

## 🛠️ Installation & Dependency Setup

Clone this repository and download dependencies using `pip`:

```bash
git clone https://github.com<your-username>/YOLOv12-Tomato-Disease-Detection.git
cd YOLOv12-Tomato-Disease-Detection
pip install -r requirements.txt
```
*Note: Dependencies require `ultralytics>=8.4.21`, `matplotlib`, and `Pillow`.*

---

## 📊 Dataset Acquisition Information
The data utilized for training this computer vision system is hosted on Kaggle:
👉 **[Kaggle Tomato Leaf Diseases Detection Dataset](https://www.kaggle.com/datasets/farukalam/tomato-leaf-diseasesdetection-computer-vision)**

Due to GitHub repository limits, raw images are not tracked here. To set up your local development pipeline:
1. Download the dataset from the Kaggle link above.
2. Structure your data directories locally according to the configuration setup inside the pipeline notebook.

### Class Annotations (7 Target Categories):
* `Bacterial Spot`
* `Early Blight`
* `Healthy`
* `Late Blight`
* `Leaf Mold`
* `Target Spot`
* `Black Spot`

---

## 📈 Experimental Evaluation & Quantitative Results

### 1. Model Computational Footprint
* **Total Network Parameters:** 26,394,501 (YOLOv12l)
* **Computational Complexity:** 89.4 GFLOPs
* **Total Layers (Pre-fusion):** 488 layers

### 2. Learning Rate Grid Tuning Peak Metrics

| Initial Learning Rate | Validation mAP@0.5 | Validation mAP@0.5:0.95 | Precision | Recall |
| :---: | :---: | :---: | :---: | :---: |
| 0.1 | *Gradient Instability* | — | — | — |
| 0.01 | 0.7010 | — | — | — |
| 0.001 | 0.6900 | — | — | — |
| 0.0001 | 0.7380 | — | — | — |
| **0.00001 (Optimal)** | **0.7620** | **0.5020** | **0.7870** | **0.7230** |

### 3. Per-Class Localisation Precision (LR = 0.00001)

| Disease Class Category | Precision | Recall | mAP@0.5 |
| :--- | :---: | :---: | :---: |
| **Healthy** | 0.972 | 1.000 | 0.995 |
| **Late Blight** | 0.838 | 0.891 | 0.935 |
| **Early Blight** | 0.769 | 0.844 | 0.875 |
| **Bacterial Spot** | 1.000 | 0.631 | 0.747 |
| **Target Spot** | 0.660 | 0.667 | 0.744 |
| **Black Spot** | 0.775 | 0.457 | 0.548 |
| **Leaf Mold** | 0.492 | 0.571 | 0.489 |

---

## 🚀 Running Inference Pipeline

To run inference on your validation sample data or custom target imagery files, navigate to the source directory and run `inference.py` after editing its `WEIGHTS_PATH` and `SOURCE` variables:

```bash
python src/inference.py
```

---

## 🏛️ Acknowledgments & Citation
This research project was conducted within the **Department of Mechatronics Engineering, School of Electrical Engineering and Technology (SEET), Federal University of Technology, Minna, Nigeria**, under the academic guidance of **Engr. Dr. Jibril Abdullahi Bala** for the course *MCE 531 – Machine Vision*.
```text
@techreport{group8_mce531_2026,
  author      = {Gideon, T. J. and Yakubu, I. W. and Raiyetunbi, H. A. and Abayomi, S. G. and Muhammad, A. K. and Aniekwe, R. C. and Ameh, C. O.},
  title       = {Performance Analysis of YOLOv12 for Multi-Class Tomato Disease Detection},
  institution = {Federal University of Technology, Minna},
  year        = {2026},
  type        = {Machine Vision Project Report}
}
```
