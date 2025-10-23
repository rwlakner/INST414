import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from math import pi

df = pd.read_csv("nba_2025.csv")
df = df[df['Team'] != '2TM']
df = df[df['G'] > 30]
df = df.drop('Awards', axis=1)
print(df.shape)

stats = df.drop('Rk', axis=1)
stats = stats.drop('Player', axis=1)
stats = stats.drop('Age', axis=1)
stats = stats.drop('Team', axis=1)
stats = stats.drop('Pos', axis=1)
stats = stats.fillna(0)

players = df['Player']

scaler = StandardScaler()
stats_scaled = scaler.fit_transform(stats)
stats_scaled = pd.DataFrame(stats_scaled, columns=stats.columns, index=players)

similarity_matrix = cosine_similarity(stats_scaled)
sim_df = pd.DataFrame(similarity_matrix, index=players, columns=players)

def get_top_similar(player_name, top_n=10):
    if player_name not in sim_df.index:
        raise ValueError(f"{player_name} not found in dataset.")
    
    top_players = sim_df[player_name].sort_values(ascending=False)[1:top_n+1]
    return top_players

targets = ['Stephen Curry', 'Anthony Davis', 'LeBron James']
for player in targets:
    print(f"\nTop Similar Players to {player}:\n")
    try:
        print(get_top_similar(player, 10))
    except ValueError:
        print("Player not found in dataset.")
        
