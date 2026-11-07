# Rainfall Transfer Learning for Semi-Arid Regions
## Case Study: Anantapur District, Andhra Pradesh

**Team Members:** Rekulakunta Teja Swaroop · Veerapaneni Lakshmi Sarayu · Sammit Menon  
**Affiliation:** Department of Computer Science and Engineering, Manipal Institute of Technology, Manipal

---

## 📌 Project Overview
Rainfall prediction in data-scarce semi-arid regions is challenging due to extreme class imbalance (~88% dry days) and short historical observation records. This project develops a two-stage deep transfer learning framework:
1. **Synoptic Pre-training:** Pre-trains recurrent neural networks (Gated Recurrent Unit - GRU) on 36+ years of ECMWF ERA5 reanalysis data to learn regional atmospheric dynamics.
2. **Local Fine-tuning:** Adapts representations onto ground-truth India Meteorological Department (IMD) gridded rainfall records (36 grid cells covering Anantapur district).
3. **Advanced Architectures:** Evaluates standard Transfer GRU, Two-Stage Hurdle models (binary classification + conditional regression), Spatial GRUs, and Intensity-Weighted GRUs against baselines.

---

## 📂 Repository Structure

```
├── data/
│   ├── era5_anantapur_1979_2023.nc    # ERA5 atmospheric reanalysis (1979-2023)
│   ├── imd_rain_2000_2023.nc          # IMD gridded daily rainfall observations (2000-2023)
│   ├── ml_sequences_2014_2023.npz     # Processed 7-day sliding window sequences
│   ├── class_distribution.json        # Rainfall category distribution statistics
│   └── selected_features.json         # Mutual information selected meteorological features
│
├── src/
│   ├── src_core_pipeline/             # Core modeling and evaluation modules
│   │   ├── models.py                  # PyTorch GRU/LSTM recurrent backbones
│   │   ├── windows.py                 # Sliding-window sequence generator
│   │   ├── losses.py                  # ClassWeightedHuberLoss function
│   │   ├── evaluate.py                # Hydrologic & categorical metrics (MAE, RMSE, POD, FAR, CSI)
│   │   ├── baselines.py               # Persistence & Day-of-Year climatology baselines
│   │   ├── preprocess.py              # Spatial cropping & z-score standardization
│   │   ├── features.py                # Feature selection algorithms
│   │   ├── interpret.py               # Relative Frobenius-norm weight shift calculator
│   │   ├── run_evaluation_honest.py   # Automated evaluation script
│   │   └── make_figures_honest.py     # Results figure generation
│   │
│   ├── src_advanced_models/           # Advanced model training routines
│   │   ├── train_transfer_models.py   # ERA5 pre-training & IMD fine-tuning
│   │   ├── train_two_stage_gru.py     # Two-stage hurdle GRU (classifier + amount regressor)
│   │   ├── train_spatial_gru.py       # Spatial coordinate-conditioned GRU
│   │   ├── train_weighted_gru.py      # Intensity-weighted loss training routine
│   │   ├── prepare_era5_imd_dataset.py# Raw dataset alignment
│   │   ├── prepare_ml_sequences.py    # Leakage-safe sequence builder
│   │   ├── evaluate_rainfall_baselines.py
│   │   ├── create_final_comparison_figures_v2.py
│   │   └── rebuild_final_master_comparison.py
│   │
│   └── demo_next_day_prediction.py    # Interactive next-day rainfall inference script
│
├── default.yaml                       # Training and dataset configuration
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Standard Python / ML gitignore
└── README.md                          # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
Clone the repository and install the required dependencies:
```bash
git clone <YOUR_REPOSITORY_URL>
cd Rainfall_Project_Submission
pip install -r requirements.txt
```

### 2. Dependencies
Key packages (see `requirements.txt` for exact versions):
- Python 3.9+
- `torch`
- `numpy`, `scipy`, `pandas`, `scikit-learn`
- `xarray`, `netCDF4`
- `matplotlib`, `pyyaml`, `joblib`

### 3. Running Model Training
To train the transfer learning models:
```bash
python src/src_advanced_models/train_transfer_models.py
```

To run the interactive next-day prediction demo:
```bash
python src/demo_next_day_prediction.py
```
