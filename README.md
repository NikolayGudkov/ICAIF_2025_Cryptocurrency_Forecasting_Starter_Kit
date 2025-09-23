# ICAIF 2025 Cryptocurrency Forecasting Starter Kit

Welcome to our **ICAIF 2025 competition: Cryptocurrency Forecasting** Starter Kit repository.  
This repository provides a standard pipeline to help you quickly get started with the competition.

For more details about the competition, visit the [Competition Website](https://hackathon.deepintomlf.ai/competitions/XX).

---

## Pipeline Overview
In this starter kit, we include:

1. **Data importing** (`train.pkl`, `x_test.pkl`, `y_test_local.pkl`)  
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
conda create -n crypto_forecast python=3.10
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
    * `time_step`: minute index within the window (ranging from 0 to 59)
    * `close`: closing price 
    * `volume`: traded volume

  * [`y_test_local.pkl`](data/): Provided to participants for local test (future 10 steps of close).

**Task Definition**:
Given the last **60 minutes** of `close, volume`, forecast the next **10 minutes of close**.
Submission format is described below.

---

## Sample Submission

A minimal example is provided in `sample_submission/` with the following files (no subfolders inside the zip):

* **`submission_example.pkl`**
  Pandas DataFrame with columns:

  * `window_id` — integer ID of each forecast window
  * `time_step` — integer horizon step **0–9** (next 10 minutes)
  * `pred_close` — float predicted closing price
    **Shape requirement:** each `window_id` must appear in **exactly 10 rows** with `time_step=0..9`.

* **`model.py`**
  Defines your model and loading logic. Must expose `init_model()` that returns a ready-to-infer model (other DL frameworks are fine if the same I/O interface is respected).

* **`model_weights.pkl`**
  Serialized weights/checkpoint loadable by `model.py`.

* **`inference.py`** (optional)
  If you use extra pre/post-processing, include `generate_forecast(x_test_path)` that produces `submission.pkl`.

Please ensure your archive is named **`submission.zip`** and the forecast file inside is exactly **`submission.pkl`** (rename from the Starter Kit if needed).

---

## Quickstart

We provide [`quickstart.ipynb`](quickstart.ipynb), which demonstrates:

1. Loading the dataset (`train.pkl`, `x_test.pkl`)
2. Training a baseline model (e.g., **ARIMA**)
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
   * **PW** (Proportional-Weighting: allocate weights proportional to predicted returns, long-only by default)  
   
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
