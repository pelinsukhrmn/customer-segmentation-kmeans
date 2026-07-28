import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_samples = 300

age = np.random.randint(18, 70, n_samples)
annual_income = np.random.randint(15, 140, n_samples)
spending_score = np.random.randint(1, 100, n_samples)

data = pd.DataFrame({
    "age": age,
    "annual_income_k": annual_income,
    "spending_score": spending_score,
})

print("=== Customer Segmentation ===\n")
print(f"Dataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())
print("\nBasic statistics:")
print(data.describe())

X = data[["annual_income_k", "spending_score"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nEvaluating cluster counts (k=2..8) with silhouette score...")
best_k = None
best_score = -1
for k in range(2, 9):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"  k={k}  silhouette={score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k

print(f"\nBest k: {best_k} (silhouette={best_score:.4f})")

model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
data["cluster"] = model.fit_predict(X_scaled)

print("\nCluster sizes:")
print(data["cluster"].value_counts().sort_index())

print("\nCluster centers (annual_income_k, spending_score):")
centers = scaler.inverse_transform(model.cluster_centers_)
for i, center in enumerate(centers):
    print(f"  Cluster {i}: income={center[0]:.1f}k, spending_score={center[1]:.1f}")

sample_customer = pd.DataFrame([{"annual_income_k": 80, "spending_score": 75}])
sample_scaled = scaler.transform(sample_customer)
predicted_cluster = model.predict(sample_scaled)[0]

print(f"\nSample Customer Prediction:")
print(f"  Income: 80k | Spending Score: 75")
print(f"  Assigned Cluster: {predicted_cluster}")

print("\nDone.")
