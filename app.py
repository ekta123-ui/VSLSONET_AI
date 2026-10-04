from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from src.analytics import (
    get_agent_stats,
    get_agent_synergy,
    get_agents,
    get_agent_comparison,
    get_best_agents_for_map,
    get_best_team_compositions,
    get_dataset_summary,
    get_map_agent_stats,
    get_map_stats,
    get_maps,
    normalize_name,
    get_roles,
)
from src.sentiment import get_score_distribution, get_sentiment_distribution
from src.review_analysis import (
    analyze_review_context,
    get_agent_catalog,
    get_agent_review_sentiment,
    get_sentiment_by_agent,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__)


def clean_review(text: str) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_assets():
    svm_path = MODELS_DIR / "svm_model.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    lstm_path = MODELS_DIR / "lstm_model.keras"
    tokenizer_path = MODELS_DIR / "tokenizer.pkl"
    label_encoder_path = MODELS_DIR / "label_encoder.pkl"

    missing = [
        str(path)
        for path in [svm_path, vectorizer_path, lstm_path, tokenizer_path, label_encoder_path]
        if not path.exists()
    ]

    if missing:
        return None

    import joblib

    svm_model = joblib.load(svm_path)
    vectorizer = joblib.load(vectorizer_path)
    tokenizer = joblib.load(tokenizer_path)
    label_encoder = joblib.load(label_encoder_path)
    lstm_model = tf.keras.models.load_model(lstm_path)

    return {
        "svm_model": svm_model,
        "vectorizer": vectorizer,
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
        "lstm_model": lstm_model,
    }


def enrich_review_context(review: str) -> dict:
    try:
        analysis = analyze_review_context(review)
    except (FileNotFoundError, ValueError) as error:
        return {
            "agents": [],
            "maps": [],
            "topics": {},
            "agent_detection_message": "Agent catalog is unavailable.",
            "map_detection_message": "Map catalog is unavailable.",
            "aspect_method": f"Entity analysis unavailable: {error}",
        }
    for detected in analysis["agents"]:
        try:
            historical = get_agent_stats(detected["agent"])
        except FileNotFoundError:
            historical = None
        detected["historical_performance"] = historical
        detected["performance_status"] = "available" if historical else "insufficient_data"
    return analysis


@app.route("/")
def index():
    return render_template("dashboard.html", active_page="dashboard")


@app.route("/dashboard")
@app.route("/maps")
@app.route("/agents")
@app.route("/recommend")
@app.route("/compare")
@app.route("/models")
@app.route("/about")
def analytics_page():
    active_page = request.path.strip("/") or "dashboard"
    return render_template("dashboard.html", active_page=active_page)


@app.route("/maps/<map_name>")
def map_detail_page(map_name: str):
    return render_template("dashboard.html", active_page="maps", selected_map=map_name)


@app.route("/agents/<path:agent_name>")
def agent_detail_page(agent_name: str):
    return render_template("dashboard.html", active_page="agents", selected_agent=agent_name)


@app.route("/sentiment")
def sentiment_page():
    return render_template("dashboard.html", active_page="sentiment")


@app.route("/predict", methods=["POST"])
@app.route("/api/sentiment/analyze", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True) or {}
        review = payload.get("review", "")
    except Exception:
        return jsonify({"error": "Invalid JSON payload."}), 400

    cleaned_review = clean_review(review)
    if not cleaned_review:
        return jsonify({"error": "Please enter a valid review before predicting."}), 400

    assets = load_assets()
    if assets is None:
        return jsonify({
            "error": "Models have not been trained yet. Run train_models.py to generate the required files in the models directory."
        }), 503

    svm_label = assets["svm_model"].predict(assets["vectorizer"].transform([cleaned_review]))[0]
    svm_result = {
        "label": str(svm_label),
        "confidence": None,
        "note": "SVM confidence is not reported because the model is not calibrated.",
    }

    tokenized = assets["tokenizer"].texts_to_sequences([cleaned_review])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized,
        maxlen=200,
        padding="post",
        truncating="post",
    )

    lstm_probs = assets["lstm_model"].predict(padded, verbose=0)[0]
    lstm_index = int(np.argmax(lstm_probs))
    lstm_label = assets["label_encoder"].inverse_transform([lstm_index])[0]
    lstm_confidence = float(lstm_probs[lstm_index])

    lstm_result = {
        "label": str(lstm_label),
        "confidence": round(lstm_confidence, 4),
    }

    analysis = enrich_review_context(review)
    return jsonify({
        "cleaned_review": cleaned_review,
        "svm": svm_result,
        "lstm": lstm_result,
        "selected_model": "SVM",
        "overall": {"model": "SVM", "label": str(svm_label), "confidence": None},
        "analysis": analysis,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "project": "VALOSENT AI"})


@app.route("/api/maps")
def api_maps():
    return jsonify({"maps": get_maps()})


@app.route("/api/dashboard-summary")
def api_dashboard_summary():
    return jsonify(get_dataset_summary())


@app.route("/api/sentiment-distribution")
@app.route("/api/sentiment/distribution")
def api_sentiment_distribution():
    try:
        return jsonify(get_sentiment_distribution())
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 503


@app.route("/api/sentiment/scores")
def api_sentiment_scores():
    try:
        return jsonify(get_score_distribution())
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 503


@app.route("/api/sentiment/by-agent")
def api_sentiment_by_agent():
    try:
        agents = [item for item in get_sentiment_by_agent() if item["total_mentions"]]
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 503
    return jsonify({
        "agents": agents,
        "method": "Mention-counted local polarity heuristic; not trained or human-validated.",
    })


@app.route("/api/agents")
def api_agents():
    return jsonify({"agents": get_agents()})


@app.route("/api/agent-catalog")
def api_agent_catalog():
    try:
        historical_agents = {normalize_name(item["agent"]) for item in get_agents()}
    except FileNotFoundError:
        historical_agents = set()
    try:
        agents = get_agent_catalog()
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 503
    return jsonify({
        "agents": [
            {**item, "has_match_data": normalize_name(item["agent"]) in historical_agents}
            for item in agents
        ]
    })


@app.route("/api/agents/<path:agent_name>")
def api_agent_profile(agent_name: str):
    normalized_name = normalize_name(agent_name)
    try:
        catalog_entry = next(
            (item for item in get_agent_catalog() if normalize_name(item["agent"]) == normalized_name),
            None,
        )
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 503
    if catalog_entry is None:
        return jsonify({"error": f"Unknown agent: {agent_name}"}), 404

    try:
        historical = get_agent_stats(catalog_entry["agent"])
    except FileNotFoundError:
        historical = None
    try:
        sentiment = get_agent_review_sentiment(catalog_entry["agent"])
    except (FileNotFoundError, ValueError):
        sentiment = None
    maps = historical["maps"] if historical else []
    qualified_maps = [item for item in maps if item["resolved_maps"] >= 20]
    best_maps = sorted(qualified_maps, key=lambda item: item["win_rate_lower_bound"], reverse=True)[:3]
    worst_maps = sorted(qualified_maps, key=lambda item: item["win_rate_lower_bound"])[:3]
    return jsonify({
        "agent": catalog_entry["agent"],
        "role": catalog_entry["role"],
        "description": None,
        "has_match_data": historical is not None,
        "historical": historical,
        "best_maps": best_maps,
        "worst_maps": worst_maps,
        "review_sentiment": sentiment,
    })


@app.route("/api/roles")
def api_roles():
    return jsonify({"roles": get_roles()})


@app.route("/api/map/<map_name>")
@app.route("/api/maps/<map_name>")
def api_map(map_name: str):
    result = get_map_stats(map_name)
    if result is None:
        return jsonify({"error": f"Unknown map: {map_name}"}), 404
    return jsonify(result)


@app.route("/api/agent/<path:agent_name>")
def api_agent(agent_name: str):
    result = get_agent_stats(agent_name)
    if result is None:
        return jsonify({"error": f"Unknown agent: {agent_name}"}), 404
    return jsonify(result)


@app.route("/api/agent-stats")
def api_agent_stats():
    agent_name = request.args.get("agent", "")
    result = get_agent_stats(agent_name)
    if result is None:
        return jsonify({"error": f"Unknown agent: {agent_name}"}), 404
    return jsonify(result)


@app.route("/api/recommend")
def api_recommend():
    map_name = request.args.get("map", "")
    result = get_best_agents_for_map(
        map_name,
        side=request.args.get("side", "both"),
        role=request.args.get("role"),
    )
    status_code = 404 if result["status"] == "unknown_map" else 200
    if result["status"] == "invalid_side":
        status_code = 400
    return jsonify(result), status_code


@app.route("/api/recommend/attack")
def api_recommend_attack():
    result = get_best_agents_for_map(
        request.args.get("map", ""),
        side="attack",
        role=request.args.get("role"),
    )
    status_code = 404 if result["status"] == "unknown_map" else 200
    return jsonify(result), status_code


@app.route("/api/recommend/defense")
def api_recommend_defense():
    result = get_best_agents_for_map(
        request.args.get("map", ""),
        side="defense",
        role=request.args.get("role"),
    )
    status_code = 404 if result["status"] == "unknown_map" else 200
    return jsonify(result), status_code


@app.route("/api/team-composition")
def api_team_composition():
    map_name = request.args.get("map", "")
    result = get_best_team_compositions(map_name)
    status_code = 404 if result["status"] == "unknown_map" else 200
    return jsonify(result), status_code


@app.route("/api/agent-synergy")
def api_agent_pairings():
    agent_name = request.args.get("agent", "")
    if not agent_name:
        return jsonify({"error": "The agent query parameter is required."}), 400
    result = get_agent_synergy(agent_name, map_name=request.args.get("map"))
    status_code = 404 if result["status"] == "unknown_agent" else 200
    return jsonify(result), status_code


@app.route("/api/agent-map-stats")
def api_agent_map_stats():
    map_name = request.args.get("map")
    if map_name and map_name.casefold() != "all" and not any(
        value.casefold() == map_name.casefold() for value in get_maps()
    ):
        return jsonify({"error": f"Unknown map: {map_name}"}), 404
    return jsonify({
        "map": map_name or "All",
        "role": request.args.get("role", "All"),
        "stats": get_map_agent_stats(
            map_name=request.args.get("map"),
            role=request.args.get("role"),
        ),
    })


@app.route("/api/agent-map-performance")
def api_agent_map_performance():
    map_name = request.args.get("map")
    agent_name = request.args.get("agent")
    role = request.args.get("role", "All")
    side = request.args.get("side", "both").casefold()
    if side not in {"both", "attack", "defense"}:
        return jsonify({"error": f"Unknown side: {side}"}), 400

    try:
        maps = get_maps()
        agents = get_agents()
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503

    canonical_map = None
    if map_name and map_name.casefold() != "all":
        canonical_map = next((value for value in maps if value.casefold() == map_name.casefold()), None)
        if canonical_map is None:
            return jsonify({"error": f"Unknown map: {map_name}"}), 404

    canonical_agent = None
    if agent_name:
        canonical_agent = next(
            (item["agent"] for item in agents if normalize_name(item["agent"]) == normalize_name(agent_name)),
            None,
        )
        if canonical_agent is None:
            return jsonify({"error": f"Unknown agent: {agent_name}"}), 404

    stats = get_map_agent_stats(canonical_map, role)
    if canonical_agent:
        stats = [item for item in stats if normalize_name(item["agent_name"]) == normalize_name(canonical_agent)]

    if side == "attack":
        stats.sort(key=lambda item: item["attack_acs_median"] if item["attack_acs_median"] is not None else float("-inf"), reverse=True)
        ranking_metric = "median T-side ACS"
    elif side == "defense":
        stats.sort(key=lambda item: item["defense_acs_median"] if item["defense_acs_median"] is not None else float("-inf"), reverse=True)
        ranking_metric = "median CT-side ACS"
    else:
        stats.sort(key=lambda item: item["appearances"], reverse=True)
        ranking_metric = "historical appearances"

    return jsonify({
        "map": canonical_map or "All",
        "agent": canonical_agent or "All",
        "role": role,
        "side": side,
        "ranking_metric": ranking_metric,
        "stats": stats,
    })


@app.route("/api/agent-comparison")
def api_agent_comparison():
    requested_agents = request.args.getlist("agent")
    if len(requested_agents) == 1 and "," in requested_agents[0]:
        requested_agents = [value.strip() for value in requested_agents[0].split(",") if value.strip()]
    if len(requested_agents) < 2 or len(requested_agents) > 5:
        return jsonify({"error": "Provide between two and five agent names."}), 400
    known_agents = {normalize_name(item["agent"]) for item in get_agents()}
    unknown_agents = [name for name in requested_agents if normalize_name(name) not in known_agents]
    if unknown_agents:
        return jsonify({"error": f"Unknown agents: {', '.join(unknown_agents)}"}), 404
    map_name = request.args.get("map")
    if map_name and map_name.casefold() != "all" and not any(
        value.casefold() == map_name.casefold() for value in get_maps()
    ):
        return jsonify({"error": f"Unknown map: {map_name}"}), 404
    return jsonify({
        "map": map_name or "All",
        "agents": requested_agents,
        "stats": get_agent_comparison(requested_agents, map_name),
    })


@app.route("/api/model-comparison")
def api_model_comparison():
    evaluation_path = BASE_DIR / "results" / "model_evaluation.json"
    if not evaluation_path.exists():
        return jsonify({"error": "Model evaluation results are not available yet."}), 404
    return jsonify(json.loads(evaluation_path.read_text(encoding="utf-8")))


@app.route("/api/match-model-results")
def api_match_model_results():
    evaluation_path = BASE_DIR / "results" / "match_model_evaluation.json"
    if not evaluation_path.exists():
        return jsonify({"error": "Match outcome model results are not available yet."}), 404
    return jsonify(json.loads(evaluation_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
