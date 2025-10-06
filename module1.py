import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load dataset
df = pd.read_csv('videos_clean.csv')

#Renaming columns
df.columns = [col.lower().strip() for col in df.columns]

#Using only needed columns
cols_to_use = ['platform', 'duration_sec', 'views', 'likes', 'shares', 'comments']
df = df[cols_to_use]

#Dropping missing or 0 value entries for length
df = df.dropna()
df = df[df['duration_sec'] > 0]

#Normalizing engagement metrics
df['likes_per_sec'] = df['likes'] / df['duration_sec']
df['shares_per_sec'] = df['shares'] / df['duration_sec']
df['views_per_sec'] = df['views'] / df['duration_sec']
df['comments_per_sec'] = df['comments'] / df['duration_sec']

#Graph Visuals
sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='duration_sec', y='views_per_sec', hue='platform', alpha=0.6)
plt.title("Video Length vs View Engagement")
plt.xlabel("Video Length (seconds)")
plt.ylabel("Views per Second")
plt.legend(title="Platform")
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='duration_sec', y='likes_per_sec', hue='platform', alpha=0.6)
plt.title("Video Length vs Like Engagement")
plt.xlabel("Video Length (seconds)")
plt.ylabel("Likes per Second")
plt.legend(title="Platform")
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='duration_sec', y='comments_per_sec', hue='platform', alpha=0.6)
plt.title("Video Length vs Comment Engagement")
plt.xlabel("Video Length (seconds)")
plt.ylabel("Comments per Second")
plt.legend(title="Platform")
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='duration_sec', y='shares_per_sec', hue='platform', alpha=0.6)
plt.title("Video Length vs Share Engagement")
plt.xlabel("Video Length (seconds)")
plt.ylabel("Shares per Second")
plt.legend(title="Platform")
plt.show()