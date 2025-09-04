from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import random


def get_clustered_reviews(reviews, embeddings):
    dist_matrix = cosine_distances(embeddings)
    db = DBSCAN(eps=0.3, min_samples=10, metric="precomputed").fit(dist_matrix)
    labels = db.labels_

    clusters = {}
    for review, label in zip(reviews, labels):
        if int(label) == -1:
            continue
        clusters[int(label)] = clusters.get(int(label), []) + [review]

    report = []
    for k, reviews in clusters.items():
        random_reviews = random.sample(reviews, min(20, len(reviews)))
        cluster_data = {"num_reviews": len(reviews), "random_reviews": random_reviews}
        report.append(cluster_data)

    return sorted(report, key=lambda x: x["num_reviews"], reverse=True)