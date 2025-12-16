import pandas as pd
import numpy as np
import unicodedata

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

NBA_CSV = "nba_2025.csv"
DRAFT_CSV = "draft_data.csv"

ID_COL = "Player"

FEATURE_COLS = [
    "PTS","FG","FG%","3P",
    "3P%","FT","FT%","ORB","DRB",
    "TRB","AST","STL","BLK"
]

VERSATILITY_THRESHOLD = 0.30

PROTOTYPES = {
    "Ball_Dominant_Scorer": "stephen curry",
    "Isolation_Scoring_Wing": "giannis antetokounmpo",
    "Slasher": "anthony edwards",
    "Offensive_Hub_Big": "nikola jokic",
    "Perimeter_Disruptor": "dyson daniels",
    "Defensive_Anchor": "evan mobley",
    "Two_Way_Disruptor": "amen thompson"
}

def normalize_name(name):
    if pd.isna(name):
        return name
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("utf-8")
    return name.lower().strip()

def assign_playstyle(row_vector, prototypes):
    row = np.nan_to_num(row_vector, nan=0.0)
    best_label, best_sim = None, -1
    for label, proto in prototypes.items():
        proto = np.nan_to_num(proto, nan=0.0)
        sim = cosine_similarity(row.reshape(1, -1), proto.reshape(1, -1))[0][0]
        if sim > best_sim:
            best_sim, best_label = sim, label
    return best_label

def closest_prototype(row_vector, prototypes):
    row = np.nan_to_num(row_vector, nan=0.0)
    best_label, best_sim = None, -1
    for label, proto in prototypes.items():
        proto = np.nan_to_num(proto, nan=0.0)
        sim = cosine_similarity(row.reshape(1, -1), proto.reshape(1, -1))[0][0]
        if sim > best_sim:
            best_sim, best_label = sim, label
    return best_label

def styles_above_threshold(row, class_labels, threshold):
    return [style for style in class_labels if row[style] >= threshold]

nba_df = pd.read_csv(NBA_CSV)
draft_df = pd.read_csv(DRAFT_CSV)

nba_df[ID_COL] = nba_df[ID_COL].apply(normalize_name)
draft_df[ID_COL] = draft_df[ID_COL].apply(normalize_name)

nba_df[FEATURE_COLS] = nba_df[FEATURE_COLS].fillna(nba_df[FEATURE_COLS].median())
draft_df[FEATURE_COLS] = draft_df[FEATURE_COLS].fillna(nba_df[FEATURE_COLS].median())

scaler = StandardScaler()
nba_scaled = scaler.fit_transform(nba_df[FEATURE_COLS])
draft_scaled = scaler.transform(draft_df[FEATURE_COLS])

nba_scaled_df = pd.DataFrame(nba_scaled, columns=FEATURE_COLS)
draft_scaled_df = pd.DataFrame(draft_scaled, columns=FEATURE_COLS)

prototype_vectors = {}

for label, player in PROTOTYPES.items():
    rows = nba_df[nba_df[ID_COL] == player]
    if rows.empty:
        raise ValueError(f"Prototype player not found: {player}")
    proto_scaled = scaler.transform(rows[FEATURE_COLS])
    prototype_vectors[label] = np.nanmean(proto_scaled, axis=0)

for k in prototype_vectors:
    prototype_vectors[k] = np.nan_to_num(prototype_vectors[k], nan=0.0)


nba_df["playstyle_label"] = [
    assign_playstyle(nba_scaled_df.iloc[i].values, prototype_vectors)
    for i in range(len(nba_scaled_df))
]

X = nba_scaled_df
y = nba_df["playstyle_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, model.predict(X_test)))

draft_probs = model.predict_proba(draft_scaled_df)
draft_preds = model.predict(draft_scaled_df)

draft_results = pd.DataFrame(draft_probs, columns=model.classes_)
draft_results["player_name"] = draft_df[ID_COL]
draft_results["primary_playstyle"] = draft_preds

class_labels = model.classes_

draft_results["styles_above_threshold"] = draft_results.apply(
    lambda r: styles_above_threshold(r, class_labels, VERSATILITY_THRESHOLD),
    axis=1
)

draft_results["num_styles_above_threshold"] = draft_results["styles_above_threshold"].apply(len)
draft_results["styles_above_threshold_str"] = draft_results["styles_above_threshold"].apply(
    lambda x: ", ".join(x)
)

def closest_nba_player(row_vector, nba_scaled_df, nba_names):
    row = np.nan_to_num(row_vector, nan=0.0)

    sims = cosine_similarity(
        row.reshape(1, -1),
        np.nan_to_num(nba_scaled_df.values, nan=0.0)
    )[0]

    best_idx = np.argmax(sims)
    return nba_names.iloc[best_idx]

draft_results["closest_nba_player"] = [
    closest_nba_player(
        draft_scaled_df.iloc[i].values,
        nba_scaled_df,
        nba_df[ID_COL]
    )
    for i in range(len(draft_scaled_df))
]

final_output = draft_results[
    [
        "player_name",
        "primary_playstyle",
        "num_styles_above_threshold",
        "styles_above_threshold_str",
        "closest_nba_player"
    ]
].sort_values(
    "num_styles_above_threshold",
    ascending=False
)

final_output = final_output[
    final_output["num_styles_above_threshold"] > 1
]

print("\n=== TOP VERSATILE DRAFT PROSPECTS ===")
print(final_output.head(20))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

plt.figure(figsize=(10, 8))
disp.plot(
    cmap=None,          
    xticks_rotation=45
)

plt.title("Confusion Matrix – NBA Playstyle Classification")
plt.tight_layout()
plt.show()
