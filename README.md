# README

## Project: Lightweight Deep Learning for Spatial Gene Expression Prediction

This repository contains the official implementation of our study on **spatial gene expression prediction from single H\&E histology spots** using **EfficientNet-B0** and lightweight deep learning strategies.

Our work establishes a **reproducible performance ceiling** for single-spot prediction tasks, significantly outperforming prior methods such as **BLEEP** and **ST-Net** on the **GSE240429 human liver dataset**【144†source】.

---

## 📌 Key Highlights

* **Single-spot prediction** pipeline (no contextual multi-spot input).
* **EfficientNet-B0 backbone** with full fine-tuning.
* **Morphology-Preserving Augmentation (MPA):** carefully designed augmentation strategy preserving biological structure.
* **Benchmark results:**

  * Up to **80% improvement** over BLEEP on highly expressed genes (HEG).
  * Outperformed ResNet-50 while using only **1/5 of the parameters**.
* **Exploratory analyses:**

  * Backbone comparison (ResNet, DenseNet, MobileNet, EfficientNet).
  * Augmentation comparison (ours vs. ST-Net).
  * Cluster-ID experiments.
  * Fine-tuning strategies (layer freezing, hidden layer depth).

---

## 📂 Repository Structure

```
BLEEP/
│── 19_make_patches_and_csv.py       # Patch extraction + dataset CSV generation
│── 20_train_effb0_sota_split8020.py # EfficientNet-B0 baseline training (80/20 split)
│── 21_train_backbones.py            # Backbone comparison (ResNet, DenseNet, etc.)
│── 22_search_freeze_and_hidden_cv5.py # 5-fold CV for layer freezing + hidden size search
│── 23_compare_augs.py               # Augmentation comparison (MPA vs ST-Net)
│── GSE240429_data/                  # Dataset folder (preprocessed data)
│── requirements.txt                 # Python dependencies
│── README.md                        # Project documentation
│── LICENSE                          # License file
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone [<your-repo-url>](https://github.com/Medical-AI-GSE240429/Code.git)
cd BLEEP
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset

* Dataset: [NCBI GEO: GSE240429](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240429)
* Place the preprocessed dataset in `GSE240429_data/` (TIFF files excluded).

### 4. Run patch extraction & dataset creation

```bash
python 19_make_patches_and_csv.py
```

### 5. Training & Evaluation

* **EfficientNet-B0 baseline:**

```bash
python 20_train_effb0_sota_split8020.py
```

* **Backbone comparison:**

```bash
python 21_train_backbones.py
```

* **5-fold CV + hyperparameter search:**

```bash
python 22_search_freeze_and_hidden_cv5.py
```

* **Augmentation comparison:**

```bash
python 23_compare_augs.py
```

---

## 📊 Results (Summary)

* EfficientNet-B0 achieved **r = 0.315** on HEG, significantly outperforming BLEEP and ST-Net【144†source】.
* Predicted **50 genes with r ≥ 0.30**, compared to only 20 by ResNet-50 with 5× more parameters.
* MPA augmentation provided stable gains compared to ST-Net augmentations.

---

## 📜 Citation

If you use this repository, please cite:

```
@article{YourPaper2025,
  title   = {Lightweight Deep Learning for Spatial Gene Expression Prediction from Single H&E Spots with EfficientNet},
  author  = {First Author and Second Author and Third Author},
  journal = {Bioinformatics / Computational Pathology},
  year    = {2025}
}
```

---

## 📌 Notes

* **Why "BLEEP"?**
  BLEEP was the previous state-of-the-art single-spot model (contrastive learning approach).
  Our work **outperforms BLEEP**, but the repo name `BLEEP` was initially kept for continuity.
  You may rename it to something more reflective (e.g., **EffB0-ST**, **HE2Gene-Lite**).

---

## 🧑‍💻 Authors

* First Author (corresponding) – *[your.email@example.com](mailto:your.email@example.com)*
* Second Author
* Third Author

Contributions follow the paper: †Equal contribution.

---

## 📄 License

This project is licensed under the terms of the **MIT License**.
