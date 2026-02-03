# Osteo-Geometric  
### Advanced Geometric Feature Engineering for Osteoporosis Screening from Knee X-Rays

---

##  Overview

Osteoporosis is a progressive skeletal disorder characterized by reduced bone mass and degradation of trabecular micro-architecture, significantly increasing fracture risk. Although Dual-Energy X-ray Absorptiometry (DEXA) is the clinical gold standard, its cost, radiation exposure, and limited availability restrict large-scale screening—especially in resource-constrained regions.

**Osteo-Geometric** proposes a lightweight, interpretable, and computation-efficient alternative using standard 2D Knee X-ray images. Instead of deep learning black boxes, this project introduces novel geometric and fractal feature engineering strategies that explicitly model trabecular texture and spatial bone organization.

The system performs **multi-class classification** into:

- Normal  
- Osteopenia (early bone loss)  
- Osteoporosis (advanced bone degradation)

---

##  Key Contributions

-  Introduces three geometrically interpretable feature extraction frameworks  
-  Captures multi-scale trabecular texture patterns without CNNs  
-  Achieves competitive performance using classical ML models  
-  Computationally efficient and suitable for clinical screening pipelines  
-  Fully explainable features grounded in geometry, fractals, and signal theory  

---

##  Dataset

- **Total Images:** 616 Knee X-ray scans  
- **Class Distribution:**
  - Normal: 221  
  - Osteopenia: 154  
  - Osteoporosis: 241  
- **Ground Truth:** Radiological assessment  

### Challenges Addressed
- Variability in knee orientation  
- Non-uniform illumination  
- Anatomical asymmetry  
- Soft-tissue interference  

---

##  Preprocessing Pipeline

To ensure geometric stability and fair feature comparison, all images undergo a standardized preprocessing workflow:

### White Pixel Suppression
- Pixels with intensity ≥ 240 (over-exposed artifacts) are set to 0.

### Square Padding
- Images are symmetrically padded to preserve anatomical centering and aspect ratio.

### Resolution Normalization
- **BCT & Sierpiński:** 512 × 512 (LANCZOS resampling)  
- **Fibonacci:** 289 × 289 (aligned with max Fibonacci radius F = 144)

---

##  Feature Engineering Methodologies

###  Balanced Cut Tree (BCT) Partitioning

**Purpose:** Capture global spatial intensity variations while remaining robust to background noise.

**Method:**
- Hierarchical, deterministic spatial decomposition  
- Proportional X-axis splits followed by uniform Y-axis partitions  
- Non-overlapping rectangular regions  

**Extracted Features (per region):**
- Mean  
- Median  
- Standard Deviation  
- Skewness  
- Kurtosis  

**Highlights:**
- Compact feature vector (~56 features)  
- Robust to uneven anatomy and padding  
- Low computational cost  

---

###  Sierpiński Triangle Fractal Masking

**Why Fractals?**  
Trabecular bone exhibits self-similar and scale-invariant texture, making fractal geometry ideal for modeling osteoporosis-related degradation.

**Method:**
- A regular hexagon is centered on the knee joint  
- Recursive Level-3 Sierpiński triangle generation  
- 78 stable triangular binary masks  

**Extracted Features (per mask):**
- Mean  
- Median  
- Standard Deviation  
- Skewness  
- Kurtosis  

**Highlights:**
- High-dimensional feature space (≈391 features)  
- Excellent sensitivity to subtle trabecular texture loss  
- Strong performance with margin-based classifiers  

**How Sierpiński Fractals Help:**  

They isolate repeating triangular micro-patterns, allowing the model to quantify texture complexity, porosity, and heterogeneity, which are hallmark indicators of osteoporosis.

---

###  Fibonacci Concentric Square Partitioning

**Motivation:** Bone degradation progresses radially—from central joint space to outer cortical bone.

**Method:**
- Concentric square rings generated using Fibonacci radii  
- Non-overlapping ring regions:
**Extracted Features (10 per region):**
-  Median Intensity  
-  Intensity Variance  
-  GLCM Homogeneity  
-  GLRLM Long Run Emphasis  
-  GLSZM Zone Percentage  
-  DCT DC Energy  
-  Permutation Entropy  
-  Laplacian Variance  
-  Compactness  
-  Approximate Lyapunov Exponent 

**Highlights:**
- 90-dimensional balanced feature vector  
- Integrates texture, frequency, entropy, and chaos theory  
- Excellent stability across train-test splits  

**Role of Fibonacci Sequence:**  

The Golden Ratio ensures natural multi-scale sampling, mirroring biological growth patterns and enabling consistent center-to-periphery analysis.

---

##  Machine Learning Models

The extracted features are evaluated using classical supervised learning algorithms:

- **SVM:** Linear, Polynomial, RBF, Sigmoid  
- **Ensemble Models:** Random Forest, Gradient Boosting  
- Logistic Regression  
- k-Nearest Neighbors  
- Decision Tree  
- Gaussian Naive Bayes  

### Experimental Setup
- **Validation:** Stratified splits (60/40, 70/30, 80/20, 90/10)  
- **Scaling:** StandardScaler for SVM, KNN, LR  
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC  

---
##  Libraries & Frameworks Used

### Programming Language

- **Python 3.x** – 3.12.3 version is used

### Image Processing & Computer Vision

- **OpenCV (cv2)** – Image manipulation, masking, filtering
- **Pillow (PIL)** – Image loading, resizing, padding

### Numerical Computing & Data Handling
- **NumPy** – Matrix operations and numerical computation
- **Pandas** – Feature matrix construction and dataset handling

### Machine Learning

- **Scikit-Learn**
- Classical ML models (SVM, Random Forest, Gradient Boosting, etc.)
- Feature scaling (StandardScaler)
- Model evaluation metrics
- Train-test splitting

### Statistical Analysis

- **SciPy**
- Skewness and kurtosis computation
- Statistical stability handling

### Texture & Signal Analysis

- **scikit-image** (where applicable)
- GLCM, GLRLM, GLSZM texture descriptors
- **NumPy FFT / DCT utilities** – Frequency-domain features

### Visualization
- **Matplotlib**
- Mask visualization
- ROC curves
- Feature distribution plots

### Parallel Processing
- **concurrent.futures (ProcessPoolExecutor)**
- Parallelized feature extraction for large datasets

---

##  Results & Observations

| Feature Method | Best Performing Models | Key Insight |
|---------------|-----------------------|-------------|
| BCT | Random Forest, SVM-Linear | Efficient spatial summarization |
| Sierpiński | SVM (RBF), Gradient Boosting | Strong trabecular texture discrimination |
| Fibonacci | Random Forest, Gradient Boosting | Most stable and balanced performance |

**Overall Best Performance:** - **Fibonacci Concentric Squares + Ensemble Models**  

This combination consistently achieved the highest precision-recall balance and robustness across splits.

---




