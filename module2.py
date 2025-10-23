import networkx as nx
import json
import matplotlib.pyplot as plt
from collections import defaultdict

with open("ml_contributors_by_repo.json", "r") as in_file:
    data = json.load(in_file)

G = nx.Graph()

for repo, contributors in data.items():
    users = [c["login"] for c in contributors if c.get("login")]
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u1, u2 = users[i], users[j]
            if G.has_edge(u1, u2):
                G[u1][u2]["weight"] += 1
            else:
                G.add_edge(u1, u2, weight=1)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

def top_n(dictionary, n=5):
    return sorted(dictionary.items(), key=lambda x: x[1], reverse=True)[:n]

print("\nTop By Degree Centrality:")
for user, score in top_n(degree_centrality):
    print(f"{user}: {score:.4f}")

print("\nTop By Eigenvector Centrality:")
for user, score in top_n(eigenvector_centrality):
    print(f"{user}: {score:.4f}")

dc_users = [user for user, _ in top_n(degree_centrality, 500)]
dc_subgraph = G.subgraph(dc_users)

ev_users = [user for user, _ in top_n(eigenvector_centrality, 500)]
ev_subgraph = G.subgraph(ev_users)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(dc_subgraph, seed=42)
nx.draw_networkx(dc_subgraph, pos, with_labels=True, node_color='skyblue', edge_color='gray')
plt.title("Top Contributors Network (by Degree Centrality)")
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(ev_subgraph, seed=42)
nx.draw_networkx(ev_subgraph, pos, with_labels=True, node_color='skyblue', edge_color='gray')
plt.title("Top Contributors Network (by Eigenvector Centrality)")
plt.axis('off')
plt.tight_layout()
plt.show()
