# DFAN: Dual-Path Fuzzy Attention Network for Time Series Forecasting

## Publication

The work presented in this repository has been published in **IEEE Transactions on Fuzzy Systems (TFS)** as an **Early Access** article.

**Title:** DFAN: Dual-Path Fuzzy Attention Network for Time Series Forecasting

**Authors:** Zhen Mei and Tao Zhao

If you find this repository useful in your research, please consider citing our paper.

---

## 1. Environment Setup

We recommend using **conda** to create a clean Python environment (tested under **Python 3.11.13**):

```bash
conda create -n dfan python=3.11.13 -y
conda activate dfan
```

---

## 2. Installation

After activating the environment, install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## 3. Data Preparation

You can directly download the **pre-processed datasets** used in our paper from:

- [Google Drive](https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link)
- [Baidu Netdisk](https://pan.baidu.com/s/1ycq7ufOD2eFOjDkjr0BfSg?pwd=bpry)

Then place the downloaded data under the folder `./dataset`.

The folder structure should look like:

```text
dataset/
 ├── ETTh1.csv
 ├── ETTh2.csv
 ├── ETTm2.csv
 └── ...
```

---

## 4. Train & Evaluate

We provide **experiment scripts** for all benchmarks in `./scripts/multivariate_forecast`.

For example, to reproduce the **ETTh1 dataset** experiment with DFAN, simply run:

```bash
sh ./scripts/multivariate_forecast/ETTh1_script/DFAN.sh
```

---

## Acknowledgement

This library is constructed based on the following repositories:

- [Time Series Forecasting Benchmark (TFB)](https://github.com/decisionintelligence/TFB)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)

We sincerely thank the authors for making their code publicly available.
