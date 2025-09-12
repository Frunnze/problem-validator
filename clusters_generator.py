from sklearn.cluster import DBSCAN, KMeans
import random
from sklearn.metrics import silhouette_score
from copy import deepcopy
import json


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
    db = DBSCAN(eps=0.3, min_samples=20, metric="cosine").fit(embeddings)
    labels = db.labels_
    return get_report(reviews, labels)


def find_optimal_k(embeddings, max_k=20):
    k_range = range(2, min(max_k + 1, len(embeddings)//10))
    
    # Early exit if range is empty or too small
    if len(k_range) == 0:
        return 2
    if len(k_range) == 1:
        return k_range[0]
    
    best_score = -1
    best_k = k_range[0]
    
    for k in k_range:
        # Reduced n_init for faster computation
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(embeddings)
        
        # Only calculate what we need
        score = silhouette_score(embeddings, labels)
        
        # Track best during iteration
        if score > best_score:
            best_score = score
            best_k = k
    
    return best_k


def get_clustered_reviews_kmeans(reviews, embeddings):
    optimal_k = find_optimal_k(embeddings)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return get_report(reviews, labels)


def get_clustered_ranked_ideas(embeddings, metadatas):
    db = DBSCAN(eps=0.3, min_samples=10, metric="cosine").fit(embeddings)
    labels = db.labels_

    clusters = {}
    for label, meta in zip(labels, metadatas):
        if int(label) == -1:
            continue
        if int(label) not in clusters:
            clusters[int(label)] = deepcopy(meta)
            clusters[int(label)]["name"] = []
        if len(clusters[int(label)]["name"]) < 10:
            clusters[int(label)]["name"].append(meta.get("name"))
        clusters[int(label)]["sum_rating"] = clusters[int(label)].get("sum_rating", 0) + meta["rating"]
        clusters[int(label)]["sim_apps_num"] = clusters[int(label)].get("sim_apps_num", 0) + 1
        clusters[int(label)]["avg_rating"] = clusters[int(label)]["sum_rating"]/clusters[int(label)]["sim_apps_num"]

    clusters = sorted(list(clusters.values()), key=lambda x: x["avg_rating"])
    print(json.dumps(clusters, indent=4))