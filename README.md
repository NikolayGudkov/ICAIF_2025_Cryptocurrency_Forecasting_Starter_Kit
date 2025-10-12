# ICAIF 2025 Cryptocurrency Forecasting Starter Kit





Welcome to the **ACM ICAIF 2025 Cryptocurrency Forecasting Competition** Starter Kit repository. It provides a standard pipeline to help you quickly get started with the competition. For more details about the competition, please refer to the [Competition Website](https://hackathon2.deepintomlf.ai/competitions/93/).



## Pipeline Overview

In this starter kit, we include:

1. **Dataset input** ( `./data/`)  
2. **A simple baseline** (`./src/baselines/`)
3. **Training & inference pipelines** (`quickstart.ipynb`)  
4. **Evaluation** (`./src/metrics.py`)  
5. **Trading strategies** (`./src/trading.py`)



## Environment Setup

We recommend creating a new Python virtual environment by **Python 3.10** and **PyTorch 2.6**. To install the required packages, run:

```console
# Create conda environment
conda create -n crypto_forecast python=3.10
conda activate crypto_forecast

# Install PyTorch (choose CUDA version as needed)
conda install pytorch torchvision torchaudio -c pytorch

# Install other requirements
pip install -r requirements.txt
```


## Data

Dataset can be obtained in two ways:

1. From the [Competition Homepage Files Phase](https://hackathon2.deepintomlf.ai/competitions/93/)  
   Download the `inputdata` package.

2. Using Git LFS:
   ```bash
   git lfs install
   git clone https://github.com/MilleXi/ICAIF_2025_Cryptocurrency_Forecasting_Starter_Kit.git
   git lfs pull
   ```

Data integrity can be verified using data/dataset_info.json.


For this challenge, the training and test data are located at [./data/](./data/).

* **Training set**:

  * [`train.pkl`](data/): Contains **continuous minute-level sequences** (close, volume).
  * Each row contains:

    * `series_id`: identifier for each continuous series
    * `time_step`: minute index within the series (starting at 0 and increasing consecutively) 
    * `close`: closing price at that minute 
    * `volume`: traded volume at that minute  

  > To create training samples, participants should extract rolling windows of **60-minute inputs** with corresponding **10-minute targets** from each `series_id` segment.

* **Test set**:

  * [`x_test.pkl`](data/): Contains sliding windows of **60-minute input sequences** (close, volume).
  * Each row has: 

    * `window_id`: identifier for each window 
    * `time_step`: minute index within the window (ranging from 0 to 59)
    * `close`: closing price at that minute
    * `volume`: traded volume at that minute

  * [`y_test_local.pkl`](data/): Contains local test data (future 10 steps of close).

**Task Definition**:
Given the last **60 minutes** of `close, volume`, forecast the next **10 minutes of close**. Submission format is described below.



## Sample Submission

A minimal submission sample is provided in [./sample\_submission/](./sample_submission/) with the following files (no subfolders inside the zip):

* **`submission_example.pkl`** is a dataframe with the following columns:
  * `window_id`: integer ID of each forecast window
  * `time_step`: integer horizon step 0-9 (next 10 minutes)
  * `pred_close`: float predicted closing price
  * **Shape requirement**: each `window_id` must appear in exactly 10 rows with `time_step=0..9`.

* **`model.py`**: Defines your model and loading logic. Must expose `init_model()` that returns a ready-to-infer model (other DL frameworks are fine if the same I/O interface is respected).

* **`model_weights.pkl`**: Serialized weights/checkpoint loadable by `model.py`.

* **`inference.py`** (optional): If you use extra pre/post-processing, include `generate_forecast(x_test_path)` that produces `submission.pkl`.

Please ensure your archive is named **`submission.zip`** and the forecast file inside is exactly **`submission.pkl`** (rename from the Starter Kit if needed).





## Quickstart

We provide a [`quickstart.ipynb`](quickstart.ipynb), which demonstrates:

1. Loading the dataset (`train.pkl`, `x_test.pkl`)
2. Training a baseline model (e.g., ARIMA)
3. Running inference to generate a `submission.pkl`

Example submission from the notebook:

```python
submission = pd.DataFrame({
    "window_id": [0,0,0,1,1,1],
    "time_step": [0,1,2,0,1,2],
    "pred_close": [1.23,1.25,1.27,0.98,0.97,0.96]
})
```



## Evaluation

Models will be assessed across four key dimensions to ensure a comprehensive evaluation: forecasting accuracy, rank-ordering capability, simulated trading profitability, and risk management. Please refer to `src/metrics.py` for details.

In addition, we also simulate performance under three representative trading strategies in `src/trading.py` to compute the profitability and risk.

* **CSM** (Cross-Sectional Momentum: long top decile, short bottom decile)
* **LOTQ** (Long-Only Top Quantile: long-only top 20%)
* **PW** (Proportional-Weighting: allocate weights proportional to predicted returns, long-only by default)  



## Citation

Please consider citing our work if you use this starter kit or related data in your work:

```
@misc{ang2025cryptoforecast,
  author       = {Y. Ang and Q. Wang and Y. Bao and X. Xi and A. K. H. Tung and Q. Huang and H. Ni and L. Szpruch},
  title        = {Cryptocurrency Forecasting Starter Kit},
  year         = {2025},
  howpublished = {\url{https://github.com/MilleXi/ICAIF_2025_Cryptocurrency_Forecasting_Starter_Kit}},
  note         = {GitHub repository}
}

@article{ang2025ctbench,
  title={CTBench: Cryptocurrency Time Series Generation Benchmark},
  author={Ang, Yihao and Wang, Qiang and Huang, Qiang and Bao, Yifan and Xi, Xinyu and Tung, Anthony KH and Jin, Chen and Huang, Zhiyong},
  journal={arXiv preprint arXiv:2508.02758},
  year={2025}
}
```

Good luck with the competition, and most importantly, **have fun!**
