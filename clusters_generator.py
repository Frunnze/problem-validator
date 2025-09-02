from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


def get_clustered_reviews(reviews, embeddings):
    dist_matrix = cosine_distances(embeddings)
    db = DBSCAN(eps=0.3, min_samples=2, metric="precomputed").fit(dist_matrix)
    labels = db.labels_

    clusters = {}
    for review, label in zip(reviews, labels):
        clusters[int(label)] = clusters.get(int(label), []) + [review]

    import json
    print(json.dumps(clusters, indent=4))
    return clusters