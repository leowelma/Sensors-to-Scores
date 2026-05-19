# From Sensors to Scores
## Sensor-Based Assessment of Practical Abilities Using Machine Learning Classification

<br><br><br>

**Author:** Leo Welma (0373885)  
**Program:** Utrecht University — Methodology and Statistics for the Behavioral, Biomedical and Social Sciences  
**Supervisors:** Remco Feskens (CITO), Johannes Steinrücke (University of Twente)  
**Date:** 20.05.2026  
**Candidate journal:** Computers in Human Behavior  
**FETC approval:** 25-2042  
**Contact:** l.welma@students.uu.nl  

---

### Study Design

This study implements and evaluates a machine learning pipeline that transforms raw IMU sensor data into calibrated probability scores and then evaluates them against psychometric criteria. For this purpose, the UAH-DriveSet, a publicly available driving dataset with six participants under three behavioral conditions (normal, drowsy, aggressive) on two road types (motorway, secondary), is used.

The pipeline involves:
- Sliding window segmentation and feature extraction
- Feature selection (near-zero variance + collinearity filtering + mutual information ranking)
- XGBoost, Random Forest, and Logistic Regression classification with isotonic calibration
- Leave-one-driver-out cross-validation
- Psychometric evaluation: calibration, score consistency, discriminant validity, SHAP feature importance
- Unsupervised validation: PCA, trip-level k-means clustering (k=3, k=6), within-driver clustering

All random seeds are fixed at **373885** to ensure full reproducibility.

---

### Data

The UAH-DriveSet is publicly available and must be downloaded separately.  
See `data/README_data.md` for full download and setup instructions.

**Source:** http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/

**Reference:**  
Romera, E., Bergasa, L.M., & Arroyo, R. (2016). Need data for driver behaviour analysis? Presenting the public UAH-DriveSet. *Proceedings of the 19th IEEE ITSC*, 387–392.

---

### Repository Structure

- `pipeline.py` — main pipeline
- `config.py` — paths, data loading
- `descriptives.py` — descriptive statistics functions
- `analysis.ipynb` — notebook with all analyses and figure generation
- `requirements.txt` — package requirements
- `environment.yml` — conda environment file
- `README.md` — this file
- `ETHICS.md` — ethics, privacy, and security statement
- `data/README_data.md` — instructions for downloading the UAH-DriveSet
- `output/figures/` — all generated figures (PNG)
- `output/tables/` — all generated tables (CSV)

---

## How to Reproduce Results

### Step 1 — Clone the repository
```bash
git clone https://github.com/leowelma/Sensors-to-Scores.git
cd Sensors-to-Scores
```

### Step 2 — Set up the environment

**Option A — pip:**
```bash
pip install -r requirements.txt
```

**Option B — conda:**
```bash
conda env create -f environment.yml
conda activate sensors-to-scores
```

### Step 3 — Download the data
Follow the instructions in `data/README_data.md` to download the UAH-DriveSet and place it in the correct folder.

### Step 4 — Configure paths
Open `config.py` and update `DATA_ROOT` to point to your local UAH-DriveSet folder:
```python
DATA_ROOT = Path("/your/path/to/UAH-DriveSet")
```

### Step 5 — Run the notebook
Run `config.py` first and then open `analysis.ipynb` and run all cells from top to bottom.

The notebook is structured as follows:
1. Environment setup
2. Data loading and descriptives
3. Feature extraction and selection
4. Unsupervised validation
5. Modeling pipeline
6. Psychometric evaluation

### Step 6 —  outputs
Find the outputs in the `output/` folder.

---

## Environment

Python 3.13 — see `requirements.txt` for full package list with version numbers, or `environment.yml` for the complete conda environment.

---

## Responsible for Archive
Leo Welma — l.welma@students.uu.nl

---

## AI Statement

During this thesis, generative AI (Claude Sonnet 4.5, Claude Sonnet 4.6, Anthropic) was used for text editing, grammar improvement, and coding assistance and debugging.