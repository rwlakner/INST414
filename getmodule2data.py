import requests
import time
import json
from tqdm import tqdm

GITHUB_TOKEN = ""  # Replace with your GitHub token
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

TOPIC = "machine-learning"
MAX_REPOS = 100         
MAX_CONTRIBUTORS = 1000

def fetch_repos_by_topic(topic, max_repos=MAX_REPOS):
    repos = []
    page = 1
    while len(repos) < max_repos:
        url = f"https://api.github.com/search/repositories?q=topic:{topic}&sort=stars&order=desc&per_page=100&page={page}"
        res = requests.get(url, headers=HEADERS)
        if res.status_code != 200:
            print("Error fetching repos:", res.json())
            break
        data = res.json().get("items", [])
        if not data:
            break
        repos.extend(data)
        page += 1
        time.sleep(1)
    return repos[:max_repos]


def fetch_contributors(repo_full_name, max_contributors=MAX_CONTRIBUTORS):
    contributors = []
    page = 1
    while len(contributors) < max_contributors:
        url = f"https://api.github.com/repos/{repo_full_name}/contributors?per_page=100&page={page}"
        res = requests.get(url, headers=HEADERS)
        if res.status_code != 200:
            print(f"Error fetching contributors for {repo_full_name}: {res.json()}")
            break
        data = res.json()
        if not data:
            break
        contributors.extend(data)
        page += 1
        time.sleep(0.5) 
    return contributors[:max_contributors]

def main():
    all_data = {}

    print(f"Fetching repos with topic: {TOPIC}")
    repos = fetch_repos_by_topic(TOPIC)
    print(f"Found {len(repos)} repositories.")

    for repo in tqdm(repos, desc="Fetching contributors"):
        repo_name = repo["full_name"]
        contributors = fetch_contributors(repo_name)
        all_data[repo_name] = [
            {"login": c.get("login"), "contributions": c.get("contributions")}
            for c in contributors if c.get("type") == "User"
        ]

    with open("ml_contributors_by_repo.json", "w") as f:
        json.dump(all_data, f, indent=2)

    print("Saved to ml_contributors_by_repo.json")

if __name__ == "__main__":
    main()

