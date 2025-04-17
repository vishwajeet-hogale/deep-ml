import numpy as np
def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    X = np.array(points, dtype=float)
    centroids = np.array(initial_centroids, dtype=float)

    for _ in range(max_iterations):
        labels = np.argmin(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)

        for i in range(k):
            cluster = X[labels == i]
            if len(cluster):
                centroids[i] = np.mean(cluster, axis=0)

    return [tuple(map(float, c)) for c in centroids]