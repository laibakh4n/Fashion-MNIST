====================================================
EE 439: Introduction to Machine Learning - Phase 2
Fashion-MNIST Classification Project
====================================================

Members:
  - Laiba Khan     (2023-EE-102)
  - Ayesha Zulfiqar (2023-EE-101)


----------------------------------------------------
PROJECT OVERVIEW
----------------------------------------------------
This project implements and evaluates three machine learning models
on the Fashion-MNIST dataset (10-class garment classification).

  Baseline Model  : Softmax / Logistic Regression (from scratch) - Laiba Khan
  Main Model      : SVM with RBF Kernel (from scratch)           - Ayesha Zulfiqar
  Flexible Model  : MLP via scikit-learn                         - Laiba Khan


----------------------------------------------------
FILES
----------------------------------------------------
  phase2.ipynb   - Main Jupyter notebook (Google Colab)
  phase2.py      - Converted .py version of the notebook
  README.txt     - This file


----------------------------------------------------
PLATFORM
----------------------------------------------------
  Environment : Google Colab (recommended)
  Python      : 3.10+
  Hardware    : Colab CPU runtime (~12 GB RAM)

  NOTE: The SVM chunked implementation is memory-aware.
  It uses on-the-fly kernel row computation (O(m) memory
  instead of O(m^2)) so it runs within ~12 GB RAM.


----------------------------------------------------
DATASET SETUP
----------------------------------------------------
Download Fashion-MNIST CSV files from:
  https://www.kaggle.com/datasets/zalando-research/fashionmnist

Place these files in your Google Drive at:
  /content/drive/MyDrive/Colab Notebooks/ml/

Required files:
  fashion-mnist_train.csv  (60,000 samples, label + 784 pixels)
  fashion-mnist_test.csv   (10,000 samples, held-out set)


----------------------------------------------------
LIBRARY VERSIONS
----------------------------------------------------
  numpy        1.26.x   - Core math and matrix operations
  pandas       2.x      - CSV loading
  scikit-learn 1.4.x    - PCA, MLPClassifier, metrics
  matplotlib   3.8.x    - Plotting and figures
  psutil       5.x      - RAM availability check

Install all with:
  pip install numpy pandas scikit-learn matplotlib psutil


----------------------------------------------------
HOW TO RUN
----------------------------------------------------

OPTION 1 - Google Colab (Recommended):
  1. Upload phase2.ipynb to Google Colab
  2. Mount Google Drive when prompted
  3. Place dataset CSV files at the path above
  4. Run all cells: Runtime > Run All

OPTION 2 - Local Jupyter:
  1. Install dependencies listed above
  2. Update the dataset path in the Preprocessing cell:
       data = pd.read_csv("path/to/fashion-mnist_train.csv")
  3. Run: jupyter notebook phase2.ipynb

OPTION 3 - Plain Python Script:
  1. Update the CSV path in phase2.py
  2. Run: python phase2.py


----------------------------------------------------
CODE OWNERSHIP
----------------------------------------------------
  Memory check cells         - Laiba Khan / Ayesha Zulfiqar
  Preprocessing              - Laiba Khan / Ayesha Zulfiqar
    (normalization, split, PCA 50 components)

  class SoftmaxRegression    - Laiba Khan
    (multiclass logistic regression from scratch, adapted from PS2)
  Logistic Regression        - Laiba Khan
    (training and evaluation)

  class MulticlassSVM        - Ayesha Zulfiqar
    (RBF kernel SVM from scratch on 10,000 subset, adapted from PS5)
  Radius search              - Ayesha Zulfiqar
    (validation search over {0.5, 1.0, 2.0, 5.0, 10.0, 20.0})
  class MulticlassSVMChunked - Ayesha Zulfiqar
    (full 48,000-sample SVM with on-the-fly O(m) kernel rows)

  MLPClassifier              - Laiba Khan
    (scikit-learn MLP, 256->128, ReLU, Adam, early stopping)

  Results and Figures        - Laiba Khan / Ayesha Zulfiqar
    (classification reports, confusion matrices, F1 charts,
     overfitting analysis)


----------------------------------------------------
OUTPUT FILES GENERATED
----------------------------------------------------
  confusion_matrices.png    - Confusion matrices for all 3 models
  f1_per_class.png          - Per-class F1 score comparison
  model_comparison.png      - Train/test accuracy and macro F1
  overfitting_analysis.png  - Train vs test gap for all 3 models


----------------------------------------------------
RESULTS SUMMARY
----------------------------------------------------
  Logistic Regression (Baseline) : Train 81.93%  Test 81.81%  F1 0.82
  SVM RBF Chunked (Main Model)   : Train 85.74%  Test 83.54%  F1 0.83
  MLP (Flexible)                 : Train 92.50%  Test 89.08%  F1 0.89

  All models are within the 5% overfitting threshold.

====================================================
