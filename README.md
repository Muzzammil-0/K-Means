# K-Means — Customer Segmentation on Wholesale Data

K-means clustering applied to the UCI Wholesale Customers dataset to discover natural customer segments from annual spend across product categories, without using the provided Channel labels.

## Dataset
- `fetch_openml('wholesale-customers', version=1)` — 440 rows, 6 spend features used for clustering
- Ground truth (`Channel`: Horeca vs Retail) held out and used only for post-hoc validation

## Approach
1. Standardize spend features (mandatory — raw ranges span two orders of magnitude)
2. Elbow method: plot inertia vs k for k=1..10
3. Silhouette score: plot vs k for k=2..10
4. Fit K-means at chosen k; report inertia, silhouette, cluster sizes
5. Visualize clusters in 2D using PCA projection; compare to true Channel labels
6. Adjusted Rand Index + confusion matrix vs ground truth
7. Interpret cluster centroids in original units (business-meaningful segment naming)
8. Compare k-means++ vs random initialization (20 runs each)

## Results
- [fill in your chosen k, silhouette, and ARI]
- Ground truth classes: Horeca (298) / Retail (142)

## Key Observations
- The elbow and silhouette methods disagree on the "best" k — this ambiguity is inherent to unsupervised learning; k must be justified, not proven
- K-means recovers the Channel split only partially (moderate ARI). Unsupervised clusters answer a different question than human labels: "what natural groupings exist?" vs. "how were these businesses categorized?"
- Centroid interpretation makes clusters interpretable in business terms (e.g., fresh-heavy = restaurants; grocery/detergent-heavy = retail)
- k-means++ produces lower and more stable inertia than random init — motivating its use as the default

## Stack
scikit-learn · numpy · pandas · matplotlib

## Run
```bash
pip install scikit-learn numpy pandas matplotlib
python wholesale_kmeans.py