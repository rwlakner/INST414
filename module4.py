import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("spotify_dataset.csv")

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

X = df[features].dropna()

df_clean = df.loc[X.index].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o')
plt.title("Elbow Method for Optimal k (Musical Features)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(k_range)
plt.show()

FINAL_K = 7

kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=10
)

df_clean["cluster"] = kmeans.fit_predict(X_scaled)

print("Cluster counts:")
print(df_clean["cluster"].value_counts())

cluster_profiles = df_clean.groupby("cluster")[features].mean()
print("\nCluster feature means:")
print(cluster_profiles)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_clean["PC1"] = X_pca[:,0]
df_clean["PC2"] = X_pca[:,1]

plt.figure(figsize=(8,6))

for c in range(FINAL_K):
    cluster_data = df_clean[df_clean["cluster"] == c]
    plt.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        alpha=0.6,
        label=f"Cluster {c}"
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"Spotify Track Clusters (k = {FINAL_K})")

plt.legend(title="Clusters")
plt.show()

example_cols = ["track_name", "artists"]

for c in range(FINAL_K):
    print(f"\n--- Cluster {c} Song Examples ---")
    examples = df_clean[df_clean["cluster"] == c][example_cols].head(2)
    print(examples.to_string(index=False))
