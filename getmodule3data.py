import pandas as pd
url = "https://www.basketball-reference.com/leagues/NBA_2025_per_game.html"
tables = pd.read_html(url)
df = tables[0]
df.to_csv("nba_2025.csv", index=False)