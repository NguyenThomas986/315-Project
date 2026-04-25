import numpy as np
from sklearn.cluster import KMeans


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Dunn Index = min inter-cluster distance / max intra-cluster diameter.
    Higher is better (more compact, well-separated clusters).
    """
    unique = np.unique(labels)
    centroids = np.array([X[labels == k].mean(axis=0) for k in unique])

    # Min distance between any two cluster centroids (inter-cluster)
    min_inter = np.inf
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < min_inter:
                min_inter = dist

    # Max diameter of any cluster (max intra-cluster distance between two points)
    max_intra = 0.0
    for k in unique:
        pts = X[labels == k]
        if len(pts) < 2:
            continue
        # Use max distance from centroid * 2 as diameter approximation (O(n) vs O(n^2))
        dists = np.linalg.norm(pts - centroids[k], axis=1)
        diameter = 2 * dists.max()
        if diameter > max_intra:
            max_intra = diameter

    if max_intra == 0:
        return 0.0
    return min_inter / max_intra


def find_optimal_k(X: np.ndarray, k_range: range, seed: int = 42) -> tuple[int, dict]:
    """
    Run K-means for each k in k_range and return the k with the highest Dunn Index.
    """
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        di = dunn_index(X, labels)
        results[k] = {"dunn": di, "inertia": km.inertia_, "labels": labels, "model": km}
        print(f"  k={k:2d} | Dunn Index={di:.4f} | Inertia={km.inertia_:,.1f}")

    best_k = max(results, key=lambda k: results[k]["dunn"])
    return best_k, results


def run_kmeans(X: np.ndarray, k: int, seed: int = 42) -> tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    return labels, km
