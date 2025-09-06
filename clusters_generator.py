from sklearn.cluster import DBSCAN, KMeans
import random
from sklearn.metrics import silhouette_score
import numpy as np


def get_report(reviews, labels):
    clusters = {}
    for review, label in zip(reviews, labels):
        if int(label) == -1:
            continue
        clusters[int(label)] = clusters.get(int(label), []) + [review]

    report = []
    for k, revs in clusters.items():
        random_reviews = random.sample(revs, min(20, len(revs)))
        cluster_data = {
            "num_reviews": len(revs),
            "percentage": (len(revs)/len(reviews))*100,
            "random_reviews": random_reviews
        }
        report.append(cluster_data)

    return sorted(report, key=lambda x: x["num_reviews"], reverse=True)


def get_clustered_reviews(reviews, embeddings):
    db = DBSCAN(eps=0.2, min_samples=20, metric="cosine").fit(embeddings)
    labels = db.labels_
    return get_report(reviews, labels)


def find_optimal_k(embeddings, max_k=20):
    inertias = []
    silhouette_scores = []
    k_range = range(2, min(max_k, len(embeddings)//10))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, labels))
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    return optimal_k


def get_clustered_reviews_kmeans(reviews, embeddings):
    optimal_k = find_optimal_k(embeddings)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return get_report(reviews, labels)