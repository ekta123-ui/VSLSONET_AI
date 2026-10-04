from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_MATCHES_PATH = BASE_DIR / "data" / "raw" / "matches" / "valorant_pro_matches.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "matches"
CLEAN_MATCHES_PATH = PROCESSED_DIR / "valorant_pro_matches_clean.parquet"
QUALITY_REPORT_PATH = PROCESSED_DIR / "quality_report.json"

MATCH_GROUP_COLUMNS = [
    "match-datetime",
    "patch",
    "map",
    "team1",
    "team2",
    "team1-score",
    "team2-score",
]
REQUIRED_TEXT_COLUMNS = [
    "match-datetime",
    "patch",
    "map",
    "team1",
    "team2",
    "player-name",
    "player-team",
    "agent",
]
NUMERIC_COLUMNS = [
    "team1-score",
    "team2-score",
    "rating",
    "rating-t",
    "rating-ct",
    "acs",
    "acs-t",
    "acs-ct",
    "k",
    "k-t",
    "k-ct",
    "d",
    "d-t",
    "d-ct",
    "a",
    "a-t",
    "a-ct",
    "tkmd",
    "tkmd-t",
    "tkmd-ct",
    "kast",
    "kast-t",
    "kast-ct",
    "adr",
    "adr-t",
    "adr-ct",
    "hs",
    "hs-t",
    "hs-ct",
    "fk",
    "fk-t",
    "fk-ct",
    "fd",
    "fd-t",
    "fd-ct",
    "fkmd",
    "fkmd-t",
    "fkmd-ct",
]
NON_NEGATIVE_COLUMNS = [
    column
    for column in NUMERIC_COLUMNS
    if column not in {"team1-score", "team2-score", "tkmd", "tkmd-t", "tkmd-ct", "fkmd", "fkmd-t", "fkmd-ct"}
]
PROPORTION_COLUMNS = ["kast", "kast-t", "kast-ct", "hs", "hs-t", "hs-ct"]


def load_raw_matches(path: Path = RAW_MATCHES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Match dataset not found: {path}")

    frame = pd.read_csv(path, encoding="latin-1", dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    index_columns = [
        column
        for column in frame.columns
        if not column or column.startswith("unnamed:")
    ]
    return frame.drop(columns=index_columns)


def clean_match_data(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = raw_frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    missing_columns = set(REQUIRED_TEXT_COLUMNS + MATCH_GROUP_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Match dataset is missing required columns: {sorted(missing_columns)}")

    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].str.strip()

    source_rows = len(frame)
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["match-datetime"] = pd.to_datetime(frame["match-datetime"], errors="coerce")
    required_text_valid = frame[REQUIRED_TEXT_COLUMNS].ne("").all(axis=1)
    agent_is_numeric = pd.to_numeric(frame["agent"], errors="coerce").notna()
    player_team_valid = frame["player-team"].eq(frame["team1"]) | frame["player-team"].eq(frame["team2"])
    scores_valid = frame[["team1-score", "team2-score"]].notna().all(axis=1)
    valid_source_rows = (
        required_text_valid
        & ~agent_is_numeric
        & player_team_valid
        & scores_valid
    )
    invalid_source_rows_removed = int((~valid_source_rows).sum())
    frame = frame.loc[valid_source_rows].copy()

    duplicate_rows_removed = int(frame.duplicated(keep="first").sum())
    frame = frame.drop_duplicates(keep="first").copy()

    group_sizes = frame.groupby(MATCH_GROUP_COLUMNS, dropna=False).size()
    agent_lineup_sizes = (
        frame.groupby(MATCH_GROUP_COLUMNS + ["player-team"], dropna=False)["agent"]
        .nunique()
        .unstack(fill_value=0)
    )
    balanced_lineups = (
        agent_lineup_sizes.eq(5).sum(axis=1).eq(2)
        & agent_lineup_sizes.gt(0).sum(axis=1).eq(2)
    )
    valid_group_keys = group_sizes.index[
        group_sizes.eq(10)
        & balanced_lineups.reindex(group_sizes.index, fill_value=False)
    ]
    invalid_map_groups_removed = len(group_sizes) - len(valid_group_keys)
    if len(valid_group_keys) == 0:
        raise ValueError("No complete 5v5 map groups remain after match-data validation.")
    valid_key_frame = valid_group_keys.to_frame(index=False)
    frame = frame.merge(valid_key_frame, on=MATCH_GROUP_COLUMNS, how="inner", validate="many_to_one")

    invalid_metric_values = {}
    for column in NON_NEGATIVE_COLUMNS:
        if column not in frame.columns:
            continue
        invalid = frame[column].lt(0)
        invalid_count = int(invalid.sum())
        if invalid_count:
            frame.loc[invalid, column] = float("nan")
            invalid_metric_values[column] = invalid_count

    for column in PROPORTION_COLUMNS:
        if column not in frame.columns:
            continue
        invalid = frame[column].notna() & ~frame[column].between(0.0, 1.0)
        invalid_count = int(invalid.sum())
        if invalid_count:
            frame.loc[invalid, column] = float("nan")
            invalid_metric_values[column] = invalid_metric_values.get(column, 0) + invalid_count

    group_text = frame[MATCH_GROUP_COLUMNS].astype(str).agg("\x1f".join, axis=1)
    frame["map_group_id"] = group_text.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    )
    team1_won = frame["team1-score"].gt(frame["team2-score"])
    team2_won = frame["team2-score"].gt(frame["team1-score"])
    frame["map_winner"] = ""
    frame.loc[team1_won, "map_winner"] = frame.loc[team1_won, "team1"]
    frame.loc[team2_won, "map_winner"] = frame.loc[team2_won, "team2"]
    frame["player_won_map"] = frame["player-team"].eq(frame["map_winner"]).where(
        frame["map_winner"].ne(""), pd.NA
    )

    frame = frame.sort_values(["match-datetime", "map_group_id", "player-team", "player-name"])
    metric_columns = [column for column in NUMERIC_COLUMNS if column in frame.columns]
    missing_metrics = {column: int(frame[column].isna().sum()) for column in metric_columns}
    report = {
        "dataset": "Valorant Pro Matches Statistics",
        "source_url": "https://www.kaggle.com/datasets/qualidea1217/valorant-pro-matches-since-april-2021",
        "license": "CC0 1.0 as listed by Kaggle; upstream records are stated to be scraped from VLR.gg",
        "source_updated": "2023-05-02",
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_rows": source_rows,
        "invalid_identity_or_score_rows_removed": invalid_source_rows_removed,
        "exact_duplicate_rows_removed": duplicate_rows_removed,
        "invalid_map_groups_removed": invalid_map_groups_removed,
        "clean_rows": len(frame),
        "clean_map_groups": int(frame["map_group_id"].nunique()),
        "date_min": frame["match-datetime"].min().date().isoformat(),
        "date_max": frame["match-datetime"].max().date().isoformat(),
        "maps": sorted(frame["map"].dropna().unique().tolist()),
        "agents": sorted(frame["agent"].dropna().unique().tolist()),
        "missing_metric_cells": missing_metrics,
        "invalid_metric_values_set_to_missing": invalid_metric_values,
        "limitations": [
            "The source has no event, region, or explicit match identifier.",
            "Map groups are reconstructed from date, patch, map, teams, and map scores.",
            "T/CT metrics are player performance statistics, not side win rates.",
            "This snapshot ends in April 2023 and does not cover later maps or agents.",
        ],
    }
    return frame, report


def run_pipeline() -> dict:
    raw_frame = load_raw_matches()
    cleaned_frame, report = clean_match_data(raw_frame)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_frame.to_parquet(CLEAN_MATCHES_PATH, index=False)
    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Cleaned data: {CLEAN_MATCHES_PATH}")
    print(f"Quality report: {QUALITY_REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_pipeline()