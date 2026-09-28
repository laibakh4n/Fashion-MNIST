# Fashion-MNIST Classification — Data First, Models Second

**EE 439: Introduction to Machine Learning — Spring 2026**
**University of Engineering and Technology (UET) Lahore**

A machine learning project built around a simple discipline: **observe the data before choosing a model**. Three classifiers — implemented from scratch and via scikit-learn — are trained on Fashion-MNIST and evaluated against hypotheses formed purely from exploratory data analysis.

> Observe → Hypothesize → Justify → Model → Evaluate

| | |
|---|---|
| **Members** | Laiba Khan (2023-EE-102) · Ayesha Zulfiqar (2023-EE-101) |
| **Dataset** | [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) (10-class garment classification) |
| **Best result** | MLP — **89.08%** test accuracy |

---

## Table of Contents

- [Project Philosophy](#project-philosophy)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Results](#results)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Key Findings](#key-findings)
- [Code Ownership](#code-ownership)
- [AI Disclosure](#ai-disclosure)

---

## Project Philosophy

Models were not picked out of convenience or familiarity — every modeling decision traces back to evidence gathered during exploratory analysis:

1. **Phase 0 — Exploratory Data Analysis:** understand the dataset (distributions, structure, noise, clusters) before touching any model. A strict 80/20 Development/Test split is enforced, with the Test Set kept completely sealed until final evaluation.
2. **Phase 1 — Hypothesis & Model Justification:** form a testable hypothesis about the required model complexity, backed by an evidence table linking each observation to a modeling implication.
3. **Phase 2 — Implementation & Evaluation:** implement a baseline, a main model, and a more flexible comparison model; evaluate fairly and check whether results support the hypothesis.
4. **Phase 3 — Reflection:** revisit the hypothesis, identify which observations were predictive vs. misleading, and outline what would be done differently.

## Repository Structure

```
.
├── phase2.ipynb          # Main notebook (Google Colab)
├── phase2.py             # Script version of the notebook
├── Final_Report.pdf       # Full written report (Phases 0–3)
├── README.md
└── outputs/               # Generated on run
    ├── confusion_matrices.png
    ├── f1_per_class.png
    ├── model_comparison.png
    └── overfitting_analysis.png
```

## Models

Model choice follows directly from the exploratory findings: PCA showed the 10 classes are **not linearly separable** in the reduced pixel space, and average-image prototypes revealed several visually confusable class pairs (Shirt/T-shirt, Coat/Pullover) — motivating a progression from linear → kernel-nonlinear → learned-representation models.

| Role | Model | Implementation | Owner |
|---|---|---|---|
| Baseline | Softmax / Logistic Regression | NumPy, from scratch (multiclass extension of PS2) | Laiba Khan |
| Main | SVM with RBF Kernel | NumPy, from scratch, memory-efficient chunked kernel (extension of PS5) | Ayesha Zulfiqar |
| Flexible comparison | MLP (256 → 128, ReLU, Adam) | scikit-learn `MLPClassifier` | Laiba Khan |

**Preprocessing (shared across all models):** pixel normalization to [0, 1], PCA to 50 components (fit on training data only, ~86% variance retained).

**Notable engineering challenge:** a full 48,000 × 48,000 RBF kernel matrix would require ~18.4 GB of RAM. This was solved by computing kernel rows on-the-fly (O(m) memory instead of O(m²)) using `float32`, allowing the SVM to train on the full training set within a ~12 GB Colab runtime.

## Results

| Model | Train Acc. | Test Acc. | Macro F1 | Train/Test Gap |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 81.93% | 81.81% | 0.82 | 0.1% |
| SVM (RBF, chunked) (Main) | 85.74% | 83.54% | 0.83 | 2.2% |
| **MLP** (Flexible) | 92.50% | **89.08%** | **0.89** | 3.4% |

All three models stay within a 5% overfitting threshold. The largest gains over the linear baseline came on the previously-identified confusable pairs: Shirt F1 improved from 0.56 → 0.73, Pullover from 0.68 → 0.81, and Coat from 0.74 → 0.84, moving from logistic regression to MLP.

Generated figures (via `phase2.ipynb` / `phase2.py`):
- `confusion_matrices.png` — per-model confusion matrices
- `f1_per_class.png` — per-class F1 comparison across models
- `model_comparison.png` — train/test accuracy and macro F1
- `overfitting_analysis.png` — train/test gap per model

## Setup

**Platform:** Google Colab recommended (CPU runtime, ~12 GB RAM). Also runs locally.

**Requirements:**

```
numpy==1.26.*
pandas==2.*
scikit-learn==1.4.*
matplotlib==3.8.*
psutil==5.*
```

```bash
pip install numpy pandas scikit-learn matplotlib psutil
```

**Dataset:** download the Fashion-MNIST CSVs from [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist):
- `fashion-mnist_train.csv` (60,000 samples)
- `fashion-mnist_test.csv` (10,000 samples, held out)

## How to Run

**Option 1 — Google Colab (recommended)**
1. Upload `phase2.ipynb` to Colab.
2. Mount Google Drive and place the dataset CSVs at `/content/drive/MyDrive/Colab Notebooks/ml/`.
3. `Runtime > Run All`.

**Option 2 — Local Jupyter**
1. Install the dependencies above.
2. Update the CSV path in the preprocessing cell:
   ```python
   data = pd.read_csv("path/to/fashion-mnist_train.csv")
   ```
3. `jupyter notebook phase2.ipynb`

**Option 3 — Plain script**
1. Update the CSV path in `phase2.py`.
2. `python phase2.py`

## Key Findings

- **Main hypothesis (confirmed):** a purely linear model cannot resolve confusable clothing pairs; moderate nonlinear complexity helps, and excessive flexibility without care overfits (an early SVM configuration showed a 14.9% train/test gap before radius tuning fixed it).
- **Alternative hypothesis (rejected):** PCA-reduced linear classification is *not* competitive with more flexible models — the 7.27-point gap between Logistic Regression and MLP is too large to attribute to noise.
- **Most predictive Phase 0 observation:** the PCA scatter plot and average-image prototypes correctly anticipated which classes (Trouser, footwear) would be easy and which pairs (Shirt/T-shirt, Coat/Pullover) would remain hard for every model.
- **Most misleading Phase 0 observation:** the 2D PCA scatter suggested cleaner separability than the models could actually exploit, since it reflects only 2 of 50 principal components.

Full reasoning, evidence tables, and reflections are in `Final_Report.pdf`.

## Code Ownership

| Component | Owner |
|---|---|
| EDA — Illustrative Figures 1–3, Non-Illustrative Figure 4–5 | Laiba Khan |
| EDA — Illustrative Figures 4–5, Non-Illustrative Figures 1–3 | Ayesha Zulfiqar |
| `SoftmaxRegression` (baseline) | Laiba Khan |
| `MulticlassSVM` / `MulticlassSVMChunked` (main model) | Ayesha Zulfiqar |
| `MLPClassifier` setup (flexible model) | Laiba Khan |
| Results, figures, hypothesis testing | Laiba Khan & Ayesha Zulfiqar |

Full section-by-section attribution is in the report's Statement of Contribution.

## AI Disclosure

Claude (Anthropic) was used to help extend the from-scratch binary logistic regression and SVM (from earlier problem sets) into multiclass versions, design the memory-efficient chunked kernel computation, and draft/structure report text — all of which was reviewed, understood, and rewritten by the authors. See `Final_Report.pdf` for the full disclosure statement.

---

*Course project for EE 439: Introduction to Machine Learning, UET Lahore, Spring 2026.*
