# Revealing Hidden Patterns in Cancer Data: An Unsupervised Learning Approach to Misdiagnosis Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper.pdf)

> Identifying misdiagnosed cancer samples and discovering biological subtypes using unsupervised machine learning on gene expression data.

**Authors:** Yaniv Lavi Ilan, Regev Yehezkel Imra  
**Institution:** Bar-Ilan University  
**Date:** May 2025

---

## 🎯 Project Overview

This project applies unsupervised learning techniques to analyze cancer gene expression data from the UCSC Xena platform (11,060 samples across 33 cancer types). By combining clustering algorithms, dimensionality reduction, and rigorous statistical validation, we identify potentially misdiagnosed samples and discover hidden biological subtypes.

### Key Achievements

✅ **Anomalies are 13× more likely to be misdiagnosed** (K-Means clustering, Fisher's exact test: p < 7.08e-13)  
✅ **89% of Cholangiocarcinoma samples** were systematically misdiagnosed (40/45 cases)  
✅ **K-Means statistically outperforms** GMM, DBSCAN, and HDBSCAN (paired t-tests, all p < 1e-6)  
✅ **Discovered distinct subtypes** in several cancer types (e.g., 4 Breast Cancer clusters, 3 Head & Neck clusters)

---

## 📊 Dataset

**Source:** [UCSC Xena Platform](https://xenabrowser.net/)

| Dataset | Description | Dimensions |
|---------|-------------|-----------|
| **Gene Expression** | EB++ Adjusted Pan-Cancer RNASeqV2 | 11,060 samples × 20,530 genes |
| **Phenotype Labels** | TCGA Cancer Type Classifications | 33 unique cancer types |

**Preprocessing Pipeline:**
1. Removed null-containing genes
2. Retained top 10% most variant genes (1,634 genes)
3. Z-score normalization
4. PCA dimension reduction (90% variance retention → 413 dimensions)

---

## 🔬 Methodology

### Preprocessing Pipeline
- Filtered to top 10% most variant genes (20,530 → 1,634 genes)
- Z-score normalization
- PCA dimension reduction (90% variance → 413 dimensions)

### Algorithms Tested
| Algorithm | Optimal Dims | Optimal Clusters | Combined Score* |
|-----------|--------------|------------------|-----------------|
| **K-Means** | 50 | 32 | 0.604 |
| **GMM** | 50 | 32 | 0.602 |
| **DBSCAN** | 100 | 42 | 0.501 |
| **HDBSCAN** | 50 | 59 | 0.553 |

*Combined Score = 0.8 × Silhouette − 0.2 × Davies-Bouldin Index

### Statistical Validation
- **50 bootstrap iterations** on test set (60/40 split)
- **Kruskal-Wallis test** (H = 243.07, p < 1e-49) confirmed significant differences
- **Paired t-tests** showed K-Means superiority across all comparisons

### Anomaly Detection
- **K-Means:** Distance to centroid > μ + 3σ
- **GMM:** Log-likelihood < 2nd percentile  
- **HDBSCAN:** Outlier score > 98th percentile

**Key Finding:** Anomalous samples identified by K-Means were 13× more likely to be misdiagnosed (Fisher's exact test)

---

## 🔍 Main Results

### 1. Misdiagnosis Detection (K-Means)
- **17/143 anomalous samples** were misdiagnosed (11.9%)
- **113/10,917 normal samples** were misdiagnosed (1.0%)
- **Odds ratio: 12.9** (p = 7.08e-13)

**Case Study:** Cholangiocarcinoma
- 40/45 samples should be reclassified as **Liver Hepatocellular Carcinoma**
- Both originate in liver tissue, suggesting systematic diagnostic confusion

### 2. Cancer Subtype Discovery
Unsupervised clustering revealed biologically distinct subgroups within single cancer types:

- **Breast Invasive Carcinoma:** 4 distinct clusters (>95% purity each)
- **Head & Neck Squamous Cell Carcinoma:** 3 distinct clusters  
- **Acute Myeloid Leukemia:** 2 clusters with 100% separation

These findings suggest underlying molecular differences that may require different treatment approaches.

### 3. Algorithm Comparison
K-Means achieved the best performance:
- **Averaged Score:** 0.7766 ± 0.01 (vs. 0.776 GMM, 0.691 DBSCAN, 0.750 HDBSCAN)
- **Statistically significant** improvement over all alternatives (p < 1e-6)

---

## 📁 Repository Structure

```
.
├── main.py                                    # Main analysis pipeline
├── paper.pdf                                # Full research paper
├── README.md                                  # This file
│
├── data/                                      # Input datasets (not included)
│   ├── EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena
│   └── TCGA_phenotype_denseDataOnlyDownload.tsv
│
├── results/
│   ├── misdiagnosed_samples.csv              # Full list of detected misdiagnoses
│   ├── most_misdiagnosed_by_algorithm.csv    # Summary by cancer type
│   ├── statistical_results.csv               # Bootstrap and statistical tests
│   ├── all_clusters_labels.csv               # Cluster assignments for all samples
│   ├── cluster_densities.csv                 # Purity metrics per cluster
│   └── combined_bootstrap_full.csv           # Raw bootstrap evaluation results
│
├── visualizations/
│   ├── Clustering_Visualizations/            # t-SNE plots for all algorithms
│   ├── cholangiocarcinoma_highlighted/       # Misdiagnosis visualizations
│   ├── head_neck_squamous_cell_carcinoma_highlighted/
│   └── breast_invasive_carcinoma_highlighted/
│
└── scripts/                                   # Utility functions (if separate)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+
pip install numpy pandas scikit-learn umap-learn hdbscan
pip install matplotlib seaborn scipy statsmodels
```

### Installation

```bash
git clone https://github.com/YanivIlan/Unsupervised_Project.git
cd Unsupervised_Project
```

### Download Data

1. **Gene Expression Data:** [EB++ Adjusted Pan-Cancer RNASeqV2](https://api.gdc.cancer.gov/data/3586c0da-64d0-4b74-a449-5ff4d9136611)
2. **Phenotype Labels:** [TCGA Phenotype](https://api.gdc.cancer.gov/data/0fc78496-818b-4896-bd83-52db1f533c5c)

Place both files in the `data/` directory.

### Run Analysis

```bash
python main.py
```

**Pipeline Steps:**
1. Data preprocessing (variance filtering, PCA)
2. Train/test split (60/40)
3. Grid search for optimal hyperparameters
4. Clustering on test set with bootstrapping
5. Statistical comparison of algorithms
6. Anomaly detection and misdiagnosis analysis
7. Visualization generation

**Expected Runtime:** ~30-60 minutes (depends on hardware)

---

## 📈 Reproducibility

All results are reproducible with the provided code and data. Key parameters:

| Component | Configuration |
|-----------|---------------|
| Random seed | 42 (fixed across all experiments) |
| Train/test split | 60/40 |
| Bootstrap iterations | 50 |
| UMAP parameters | Grid search: 1-10, 50, 100, 150, 200 dimensions |
| Cluster range | 2-15, 20-50 (step 4) |
| Dominant threshold | 90% purity |
| Anomaly thresholds | μ + 3σ (K-Means), 2nd/98th percentile (GMM/HDBSCAN) |

---

## 🔮 Future Directions

1. **Feature Importance Analysis** - Identify specific genes driving cluster separation
2. **External Validation** - Test on independent cohorts (ICGC, other TCGA datasets)
3. **Clinical Integration** - Deploy as diagnostic quality control tool
4. **Survival Analysis** - Correlate discovered subtypes with patient outcomes

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{lavi2025unsupervised,
  author = {Lavi Ilan, Yaniv and Yehezkel Imra, Regev},
  title = {Revealing Hidden Patterns in Cancer Data: An Unsupervised Learning Approach to Misdiagnosis Detection},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/YanivIlan/Unsupervised_Project}}
}
```

---
