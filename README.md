# Customer Segmentation (KMeans)

Segments customers by income and spending behavior using KMeans clustering.

- Dataset: synthetic, generated with numpy (300 samples)
- Algorithm: KMeans

## How it works

1. Generates a synthetic dataset with `age`, `annual_income_k`, and `spending_score`.
2. Scales `annual_income_k` and `spending_score` with `StandardScaler`.
3. Tries k=2..8, scores each with silhouette score, and picks the best k.
4. Fits KMeans with that k and prints cluster sizes and centers.
5. Predicts the cluster for a sample customer (income=80k, spending_score=75).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python customer_segmentation.py
```
