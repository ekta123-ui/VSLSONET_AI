from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
CLEAN_MATCHES_PATH = BASE_DIR / "data" / "processed" / "matches" / "valorant_pro_matches_clean.parquet"
AGENT_ROLES_PATH = BASE_DIR / "data" / "agents" / "agent_roles.csv"
CONFIG_PATH = BASE_DIR / "config" / "ml_model.json"
MODEL_PATH = BASE_DIR / "models" / "analytics_model.pkl"
RESULTS_PATH = BASE_DIR / "results" / "match_model_evaluation.json"

TARGET_NAMES = ["Team 2 wins", "Team 1 wins"]
LEAKAGE_EXCLUDED_FEATURES = [
    "map scores and winner",
    "post-match rating, ACS, K/D, KAST, ADR, headshots, and first kills/deaths",
    "T-side and CT-side player metrics",
]


def build_match_records(matches: pd.DataFrame) -> pd.DataFrame:
    roles = pd.read_csv(AGENT_ROLES_PATH)
    roles["agent_key"] = roles["agent"].str.casefold()
    role_by_agent = roles.set_index("agent_key")["role"].to_dict()

    records = []
    for group_id, group in matches.groupby("map_group_id", sort=False):
        first = group.iloc[0]
        team1 = first["team1"]
        team2 = first["team2"]
        team1_agents = sorted(set(group.loc[group["player-team"].eq(team1), "agent"]))
        team2_agents = sorted(set(group.loc[group["player-team"].eq(team2), "agent"]))
        if len(team1_agents) != 5 or len(team2_agents) != 5:
            continue

        score1 = float(first["team1-score"])
        score2 = float(first["team2-score"])
        if score1 == score2:
            continue

        records.append({
            "map_group_id": group_id,
            "map": first["map"],
            "patch": str(first["patch"]),
            "team1": team1,
            "team2": team2,
            "team1_agents": team1_agents,
            "team2_agents": team2_agents,
            "team1_won": int(score1 > score2),
            "series_group": "|".join(sorted((
                first["match-datetime"].date().isoformat(),
                team1,
                team2,
            ))),
        })

    return pd.DataFrame.from_records(records)


def _feature_columns(agents: list[str], roles: list[str]) -> tuple[list[str], list[str]]:
    categorical_columns = ["map", "patch"]
    indicator_columns = [
        f"{side}_{kind}_{value}"
        for side in ("team1", "team2")
        for kind, values in (("agent", agents), ("role", roles))
        for value in values
    ]
    return categorical_columns, indicator_columns


def make_feature_frame(
    rows: pd.DataFrame,
    agents: list[str],
    roles: list[str],
    role_by_agent: dict[str, str],
) -> pd.DataFrame:
    features = pd.DataFrame({"map": rows["map"], "patch": rows["patch"]}, index=rows.index)
    for side in ("team1", "team2"):
        side_agents = rows[f"{side}_agents"]
        for agent in agents:
            features[f"{side}_agent_{agent}"] = side_agents.map(lambda lineup: int(agent in lineup))
        for role in roles:
            features[f"{side}_role_{role}"] = side_agents.map(
                lambda lineup: sum(role_by_agent.get(agent.casefold()) == role for agent in lineup)
            )
    return features


def _mirror_training_rows(
    rows: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    mirrored = rows.copy()
    mirrored["team1_agents"], mirrored["team2_agents"] = (
        rows["team2_agents"].copy(),
        rows["team1_agents"].copy(),
    )
    mirrored_features = features.copy()
    mirrored_features["map"] = rows["map"].values
    mirrored_features["patch"] = rows["patch"].values

    feature_names = [column for column in features.columns if column not in {"map", "patch"}]
    for column in feature_names:
        other_side = "team2" if column.startswith("team1_") else "team1"
        suffix = column.split("_", 1)[1]
        mirrored_features[column] = features[f"{other_side}_{suffix}"].values

    labels = pd.concat([
        rows["team1_won"].astype(int),
        1 - rows["team1_won"].astype(int),
    ], ignore_index=True)
    return pd.concat([features, mirrored_features], ignore_index=True), labels


def _build_pipeline(categorical_columns: list[str], numeric_columns: list[str], config: dict) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("lineup", "passthrough", numeric_columns),
    ])
    model = LogisticRegression(
        max_iter=config["max_iter"],
        class_weight=config["class_weight"],
        random_state=config["random_state"],
    )
    return Pipeline([("features", preprocessor), ("model", model)])


def train_match_model() -> dict:
    if not CLEAN_MATCHES_PATH.exists():
        raise FileNotFoundError("Run `python -m src.data_loader` before training the match outcome model.")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    matches = pd.read_parquet(CLEAN_MATCHES_PATH)
    records = build_match_records(matches)
    if len(records) < config["minimum_rows"]:
        raise ValueError(f"Only {len(records)} valid map outcomes are available for training.")

    agents = sorted({agent for lineup in records["team1_agents"].tolist() + records["team2_agents"].tolist() for agent in lineup})
    roles_frame = pd.read_csv(AGENT_ROLES_PATH)
    role_by_agent = dict(zip(roles_frame["agent"].str.casefold(), roles_frame["role"]))
    roles = sorted(set(role_by_agent.values()))
    features = make_feature_frame(records, agents, roles, role_by_agent)
    categorical_columns, numeric_columns = _feature_columns(agents, roles)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config["test_size"],
        random_state=config["random_state"],
    )
    train_indices, test_indices = next(
        splitter.split(features, records["team1_won"], groups=records["series_group"])
    )
    train_rows = records.iloc[train_indices].reset_index(drop=True)
    original_train_features = features.iloc[train_indices].reset_index(drop=True)
    original_train_labels = train_rows["team1_won"].astype(int)
    test_rows = records.iloc[test_indices].reset_index(drop=True)
    test_features = features.iloc[test_indices].reset_index(drop=True)
    mirrored_features, mirrored_labels = _mirror_training_rows(train_rows, original_train_features)

    pipeline = _build_pipeline(categorical_columns, numeric_columns, config)
    start = __import__("time").perf_counter()
    pipeline.fit(mirrored_features, mirrored_labels)
    training_seconds = __import__("time").perf_counter() - start

    start = __import__("time").perf_counter()
    predictions = pipeline.predict(test_features)
    inference_ms = __import__("time").perf_counter() - start
    inference_ms_per_map = inference_ms * 1000 / len(test_rows)
    actual = test_rows["team1_won"].astype(int)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(original_train_features, original_train_labels)
    baseline_predictions = baseline.predict(test_features)

    report = classification_report(
        actual,
        predictions,
        labels=[0, 1],
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )
    baseline_report = classification_report(
        actual,
        baseline_predictions,
        labels=[0, 1],
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )
    results = {
        "model": "Logistic Regression",
        "task": "Predict team 1 map win from map, patch, and pre-match agent/role lineups",
        "source": "Valorant Pro Matches Statistics; Kaggle snapshot through 2023-04-16",
        "features": ["map", "patch", "team agent indicators", "team role counts"],
        "excluded_to_prevent_leakage": LEAKAGE_EXCLUDED_FEATURES,
        "split": "GroupShuffleSplit by match date and unordered team pair; mirrored training rows only",
        "map_outcomes": len(records),
        "series_groups": int(records["series_group"].nunique()),
        "train_map_outcomes": len(train_rows),
        "test_map_outcomes": len(test_rows),
        "training_time_seconds": training_seconds,
        "inference_latency_ms_per_map": inference_ms_per_map,
        "accuracy": accuracy_score(actual, predictions),
        "precision_macro": precision_score(actual, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(actual, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(actual, predictions, average="macro", zero_division=0),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(actual, predictions, labels=[0, 1]).tolist(),
        "baseline": {
            "model": "Most-frequent class",
            "accuracy": accuracy_score(actual, baseline_predictions),
            "f1_macro": f1_score(actual, baseline_predictions, average="macro", zero_division=0),
            "classification_report": baseline_report,
            "confusion_matrix": confusion_matrix(actual, baseline_predictions, labels=[0, 1]).tolist(),
        },
        "limitations": [
            "The source has no explicit match ID, event, region, or starting-side data.",
            "The split groups same-date team pairings to reduce series leakage but cannot guarantee event-level separation.",
            "The target is map winner; no side-specific win prediction is supported.",
            "The dataset is historical through April 2023; performance may not transfer to current patches.",
        ],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "agents": agents,
        "roles": roles,
        "role_by_agent": role_by_agent,
        "maps": sorted(records["map"].unique().tolist()),
        "patches": sorted(records["patch"].unique().tolist()),
        "feature_columns": features.columns.tolist(),
    }, MODEL_PATH)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    train_match_model()