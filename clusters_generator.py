from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


def get_clustered_reviews(reviews, embeddings):
    dist_matrix = cosine_distances(embeddings)
    db = DBSCAN(eps=0.3, min_samples=10, metric="precomputed").fit(dist_matrix)
    labels = db.labels_

    clusters = {}
    for review, label in zip(reviews, labels):
        if int(label) == -1:
            continue
        clusters[int(label)] = clusters.get(int(label), []) + [review]

    return clusters