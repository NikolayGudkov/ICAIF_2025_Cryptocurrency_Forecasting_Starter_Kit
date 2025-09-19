# ICAIF 2025 Cryptocurrency Forecasting Starter Kit

Welcome to our **ICAIF 2025 competition: Cryptocurrency Forecasting** Starter Kit repository.  
This repository provides a standard pipeline to help you quickly get started with the competition.

For more details about the competition, visit the [Competition Website](https://hackathon.deepintomlf.ai/competitions/XX).

---

## Pipeline Overview
In this starter kit, we include:

1. **Data importing** (`train.pkl`, `x_test.pkl`, `y_test.pkl` for organizers)  
2. **Model baselines** (a vae-based model)  
3. **Training & inference pipelines** (with `quickstart.ipynb`)  
4. **Evaluation module** (`src/metrics.py` for official metrics)  
5. **Trading simulation module** (`src/trading.py` for portfolio-style evaluation)

---

## Environment Setup
The code has been tested successfully using **Python 3.10** and **PyTorch 2.6**.  
We recommend creating a new Python virtual environment.

To install the required packages, run:

```console
# Create environment (example with conda)
conda create -n crypto_forecast python=3.9
conda activate crypto_forecast

# Install PyTorch (choose CUDA version as needed)
conda install pytorch torchvision torchaudio -c pytorch

# Install other requirements
pip install -r requirements.txt
```

---

## Data

For this challenge, the training and test data are located at [data/](data/).

* **Training set**:

  * [`train.pkl`](data/): Contains **continuous minute-level sequences** (close, volume).
  * Each row has:

    * `series_id`: identifier for each continuous series
    * `time_step`: minute index within the series (starting at 0 and increasing consecutively) 
    * `close`: closing price at that minute 
    * `volume`: traded volume at that minute  

  > To create training samples, participants should extract rolling windows of **60-minute inputs** with corresponding **10-minute targets** from each `series_id` segment.

* **Test set**:

  * [`x_test.pkl`](data/): Contains sliding windows of **60-minute input sequences** (close, volume).
  * Each row has: 

    * `window_id`: identifier for each training window 
    * `time_step`: 0..59 for input part 
    * `close`: closing price 
    * `volume`: traded volume

  * [`y_test.pkl`](data/): Provided only to organizers for offline evaluation (future 10 steps of close).

**Task Definition**:
Given the last **60 minutes** of `close, volume`, forecast the next **10 minutes of close**.
Submission format is described below.

---

## Sample Submission

We provide a sample submission file at [sample\_submission/](sample_submission/) which includes:

* **`submission_example.pkl`**: A DataFrame with the following columns:

  * `window_id`: ID of each forecast window
  * `time_step`: horizon step (0..9 for 10 minutes ahead)
  * `pred_close`: your predicted close price

**Shape requirement**: For each `window_id`, there must be exactly 10 rows with `time_step=0..9`.

---

## Quickstart

We provide [`quickstart.ipynb`](quickstart.ipynb), which demonstrates:

1. Loading the dataset (`train.pkl`, `x_test.pkl`)
2. Training a baseline model (e.g., **TinyTimeVAE**)
3. Running inference to generate a `submission.pkl`

Example preview from the notebook:

```python
submission = pd.DataFrame({
    "window_id": [0,0,0,1,1,1],
    "time_step": [0,1,2,0,1,2],
    "pred_close": [1.23,1.25,1.27,0.98,0.97,0.96]
})
```

---

## Evaluation

We provide two levels of evaluation:

1. **Official Metrics** (`src/metrics.py`)

   * MSE, MAE, IC, IR, SharpeRatio, MDD, VaR, ES

2. **Trading Simulation** (`src/trading.py`)

   * **CSM** (Cross-Sectional Momentum: long top decile, short bottom decile)
   * **LOTQ** (Long-Only Top Quantile: long-only top 20%)

These give both statistical accuracy and portfolio-style economic interpretation.

---

## Citation

If you use this starter kit or related data in your work, please cite:

```
Y. Ang, Q. Wang, Y. Bao, X. Xi, A. K. H. Tung, Q. Huang, H. Ni, and L. Szpruch.
https://github.com/MilleXi/ICAIF_2025_Cryptocurrency_Forecasting_Starter_Kit, 2025
```

---

Good luck with the competition, and most importantly — **have fun!**
