from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REVIEW_DATA_PATH = BASE_DIR / "data" / "user_reviews.parquet"
AGENT_ROLES_PATH = BASE_DIR / "data" / "agents" / "agent_roles.csv"
SENTIMENTS = ("Positive", "Neutral", "Negative")
POLARITY_WORDS = {
    "Positive": ("amazing", "awesome", "balanced", "best", "excellent", "effective", "enjoy", "enjoyable", "fun", "good", "great", "helpful", "love", "like", "powerful", "solid", "strong", "useful", "win"),
    "Negative": ("annoying", "awful", "bad", "broken", "buggy", "frustrating", "hate", "terrible", "weak", "worse", "worst"),
}
NEGATIONS = re.compile(r"\b(?:not|never|no|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|can't|cannot|won't)\b(?:\W+\w+){0,2}\W*$", re.IGNORECASE)
CLAUSE_BOUNDARY = re.compile(r"[,;.!?]|\b(?:but|however|although|though|yet)\b", re.IGNORECASE)
TOPIC_TERMS = {
    "Gameplay": ("gameplay", "gunplay", "ability", "abilities", "aim", "recoil", "movement", "shooting", "weapon", "weapons"),
    "Matchmaking": ("matchmaking", "match making"),
    "Patch": ("patch", "update", "nerf", "nerfed", "buff", "buffed"),
    "Performance": ("performance", "fps", "lag", "ping", "server", "hitreg", "hit registration", "frame rate"),
}


@lru_cache(maxsize=1)
def _load_review_frame() -> pd.DataFrame:
    if not REVIEW_DATA_PATH.exists():
        raise FileNotFoundError(f"Review dataset not found: {REVIEW_DATA_PATH}")

    frame = pd.read_parquet(REVIEW_DATA_PATH)
    columns = {str(column).strip().casefold(): column for column in frame.columns}
    text_column = next((columns[name] for name in ("review_text", "review", "text") if name in columns), None)
    date_column = next((columns[name] for name in ("published_date", "date", "review_date") if name in columns), None)
    if text_column is None:
        raise ValueError("Review dataset must contain a review-text column.")

    return pd.DataFrame({
        "review_text": frame[text_column].fillna("").astype(str),
        "published_date": pd.to_datetime(frame[date_column], errors="coerce") if date_column else pd.NaT,
    })


@lru_cache(maxsize=1)
def get_agent_catalog() -> list[dict]:
    if not AGENT_ROLES_PATH.exists():
        raise FileNotFoundError(f"Agent role dataset not found: {AGENT_ROLES_PATH}")

    roles = pd.read_csv(AGENT_ROLES_PATH)
    required = {"agent", "role"}
    if not required.issubset({str(column).strip().casefold() for column in roles.columns}):
        raise ValueError("Agent role dataset must contain agent and role columns.")
    roles.columns = [str(column).strip().casefold() for column in roles.columns]
    roles = roles.dropna(subset=["agent", "role"]).drop_duplicates(subset=["agent"])
    return [
        {"agent": str(row.agent).strip(), "role": str(row.role).strip()}
        for row in roles.itertuples(index=False)
    ]


@lru_cache(maxsize=1)
def _get_map_names() -> tuple[str, ...]:
    from src.analytics import get_maps

    try:
        return tuple(get_maps())
    except FileNotFoundError:
        return ()


def _name_matches(text: str, names: tuple[str, ...] | list[str]) -> list[dict]:
    matches = []
    for name in names:
        aliases = [name]
        if name.casefold() == "kay/o":
            aliases.append("Kayo")
        for alias in aliases:
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])", re.IGNORECASE)
            matches.extend({"name": name, "start": match.start(), "end": match.end()} for match in pattern.finditer(text))
    unique = {(item["name"], item["start"], item["end"]): item for item in matches}
    return sorted(unique.values(), key=lambda item: (item["start"], item["end"]))


def _clause_at(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    right = len(text)
    for boundary in CLAUSE_BOUNDARY.finditer(text):
        if boundary.end() <= start:
            left = boundary.end()
        elif boundary.start() >= end:
            right = boundary.start()
            break
    return left, right


def _polarity_for_mention(text: str, start: int, end: int) -> dict:
    left, right = _clause_at(text, start, end)
    clause = text[left:right]
    evidence = []
    for label, words in POLARITY_WORDS.items():
        for word in words:
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", re.IGNORECASE)
            for match in pattern.finditer(clause):
                polarity = label
                if NEGATIONS.search(clause[max(0, match.start() - 32):match.start()]):
                    polarity = "Negative" if label == "Positive" else "Positive"
                evidence.append({
                    "word": match.group(0),
                    "label": polarity,
                    "distance": abs(left + match.start() - start),
                })

    if not evidence:
        return {"sentiment": "Neutral", "evidence": [], "basis": "No polarity words found near the mention."}

    labels = {item["label"] for item in evidence}
    if len(labels) == 1:
        sentiment = next(iter(labels))
        relevant = evidence
    else:
        nearest_distance = min(item["distance"] for item in evidence)
        nearest = [item for item in evidence if item["distance"] == nearest_distance]
        nearest_labels = {item["label"] for item in nearest}
        sentiment = next(iter(nearest_labels)) if len(nearest_labels) == 1 else "Neutral"
        relevant = nearest

    return {
        "sentiment": sentiment,
        "evidence": list(dict.fromkeys(item["word"] for item in relevant)),
        "basis": "Transparent local polarity-word heuristic; not a trained aspect model.",
    }


def _topic_mentions(text: str, terms: tuple[str, ...]) -> list[dict]:
    matches = []
    for term in terms:
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.IGNORECASE)
        matches.extend({"start": match.start(), "end": match.end()} for match in pattern.finditer(text))
    return sorted(matches, key=lambda item: item["start"])


def analyze_review_context(review: str) -> dict:
    text = str(review or "")
    agent_names = tuple(item["agent"] for item in get_agent_catalog())
    detected_agents = []
    for mention in _name_matches(text, agent_names):
        result = _polarity_for_mention(text, mention["start"], mention["end"])
        detected_agents.append({"agent": mention["name"], **result})

    detected_maps = []
    for mention in _name_matches(text, _get_map_names()):
        result = _polarity_for_mention(text, mention["start"], mention["end"])
        detected_maps.append({"map": mention["name"], **result})

    topics = {}
    for topic, terms in TOPIC_TERMS.items():
        mentions = _topic_mentions(text, terms)
        results = [_polarity_for_mention(text, item["start"], item["end"]) for item in mentions]
        labels = {item["sentiment"] for item in results if item["evidence"]}
        sentiment = next(iter(labels)) if len(labels) == 1 else "Mixed" if labels else None
        topics[topic] = {
            "detected": bool(mentions),
            "sentiment": sentiment,
            "evidence": list(dict.fromkeys(word for item in results for word in item["evidence"])),
            "basis": "Transparent local polarity-word heuristic; not a trained aspect model.",
        }

    return {
        "agents": detected_agents,
        "maps": detected_maps,
        "topics": topics,
        "agent_detection_message": None if detected_agents else "No VALORANT agent detected in this review.",
        "map_detection_message": None if detected_maps else "No map detected.",
        "aspect_method": "Rule/context heuristic; agent- and topic-level sentiment is not human-labeled or model-validated.",
    }


@lru_cache(maxsize=1)
def get_sentiment_by_agent() -> list[dict]:
    frame = _load_review_frame()
    catalog = get_agent_catalog()
    counts = {
        item["agent"]: {label: 0 for label in SENTIMENTS}
        for item in catalog
    }
    trends: dict[str, dict[str, dict[str, int]]] = {item["agent"]: {} for item in catalog}

    for review, published_date in zip(frame["review_text"], frame["published_date"]):
        mentions = _name_matches(review, tuple(counts))
        seen = set()
        for mention in mentions:
            agent = mention["name"]
            if agent in seen:
                continue
            seen.add(agent)
            sentiment = _polarity_for_mention(review, mention["start"], mention["end"])["sentiment"]
            counts[agent][sentiment] += 1
            if pd.notna(published_date):
                date_key = published_date.date().isoformat()
                trends[agent].setdefault(date_key, {label: 0 for label in SENTIMENTS})
                trends[agent][date_key][sentiment] += 1

    rows = []
    for item in catalog:
        agent = item["agent"]
        total = sum(counts[agent].values())
        rows.append({
            **item,
            "positive_reviews": counts[agent]["Positive"],
            "neutral_reviews": counts[agent]["Neutral"],
            "negative_reviews": counts[agent]["Negative"],
            "total_mentions": total,
            "positive_percentage": round(counts[agent]["Positive"] / total * 100, 2) if total else 0.0,
            "neutral_percentage": round(counts[agent]["Neutral"] / total * 100, 2) if total else 0.0,
            "negative_percentage": round(counts[agent]["Negative"] / total * 100, 2) if total else 0.0,
            "trend": [
                {"date": date, **daily_counts}
                for date, daily_counts in sorted(trends[agent].items())
            ],
            "method": "Mention-counted local polarity heuristic; not trained or human-validated.",
        })
    return rows


def get_agent_review_sentiment(agent_name: str) -> dict | None:
    normalized = re.sub(r"[^a-z0-9]", "", str(agent_name).casefold())
    for item in get_sentiment_by_agent():
        key = re.sub(r"[^a-z0-9]", "", item["agent"].casefold())
        if key == normalized:
            return item
    return None