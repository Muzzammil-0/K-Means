import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix

# ---------- 1. Load ----------
data = fetch_openml('wholesale-customers', version=1, as_frame=True)
df = data.frame.copy()

# Ground truth
y_true = df['Channel'].astype(int).values

# Use only the 6 numeric spend features
spend_cols = ['V2', 'V3', 'V4', 'V5', 'V6', 'V7']
X = df[spend_cols].values
feature_names = spend_cols

print("Shape:", X.shape)
print("Ranges:")
for name, col in zip(feature_names, X.T):
    print(f"  {name}: min={col.min():.0f}  max={col.max():.0f}")
# ---------- 2. Scale (mandatory for K-means) ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Show why: feature ranges differ wildly
print("\nRaw feature ranges:")
for name, col in zip(feature_names, X.T):
    print(f"  {name:18s} min={col.min():8.0f}  max={col.max():8.0f}")

# ---------- 3. Elbow method ----------
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o')
plt.xlabel("k"); plt.ylabel("Inertia (within-cluster SSE)")
plt.title("Elbow Method — Wholesale Customers")
plt.grid(alpha=0.3); plt.show()

# ---------- 4. Silhouette method ----------
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sil_scores, marker='s', color='green')
plt.xlabel("k"); plt.ylabel("Silhouette score")
plt.title("Silhouette Score vs k")
plt.grid(alpha=0.3); plt.show()

# ---------- 5. Fit at chosen k ----------
k_chosen = 3   # adjust based on your elbow + silhouette plots
km = KMeans(n_clusters=k_chosen, n_init=10, random_state=42).fit(X_scaled)
labels = km.labels_

print(f"\nK-means at k={k_chosen}")
print(f"  Inertia:          {km.inertia_:.2f}")
print(f"  Silhouette score: {silhouette_score(X_scaled, labels):.4f}")
print(f"  Cluster sizes:    {np.bincount(labels)}")

# ---------- 6. Visualize via PCA (callback to your earlier project) ----------
pca = PCA(n_components=2).fit(X_scaled)
X_2d = pca.transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# K-means clusters
for c in range(k_chosen):
    axes[0].scatter(X_2d[labels == c, 0], X_2d[labels == c, 1],
                    label=f"Cluster {c}", alpha=0.6, s=30)
axes[0].set_title(f"K-means clusters (k={k_chosen})")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Ground truth
for c in np.unique(y_true):
    axes[1].scatter(X_2d[y_true == c, 0], X_2d[y_true == c, 1],
                    label=f"True class {c}", alpha=0.6, s=30)
axes[1].set_title("Ground truth (Channel)")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()

# ---------- 7. How well do clusters match the true labels? ----------
ari = adjusted_rand_score(y_true, labels)
print(f"\nAdjusted Rand Index vs ground truth: {ari:.4f}  (1.0=perfect, 0=random)")
print("\nConfusion matrix (rows=true, cols=cluster):")
print(confusion_matrix(y_true, labels))

# ---------- 8. Interpret centroids (in original units) ----------
centroids_original = scaler.inverse_transform(km.cluster_centers_)
centroids_df = pd.DataFrame(centroids_original, columns=feature_names)
centroids_df.index.name = 'Cluster'
print("\nCluster centroids (original units):")
print(centroids_df.round(0))

# ---------- 9. k-means++ vs random init ----------
print("\n--- Init comparison (k=3, 20 runs each) ---")
for init in ['k-means++', 'random']:
    scores = []
    for seed in range(20):
        km_i = KMeans(n_clusters=3, n_init=1, init=init, random_state=seed).fit(X_scaled)
        scores.append(km_i.inertia_)
    print(f"  {init:10s} inertia: mean={np.mean(scores):.1f}, std={np.std(scores):.1f}, min={np.min(scores):.1f}, max={np.max(scores):.1f}")