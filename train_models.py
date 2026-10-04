from __future__ import annotations

import csv
import gc
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from src.sentiment import SENTIMENT_LABELS, create_sentiment

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MAX_LENGTH = 200
LABELS = list(SENTIMENT_LABELS)


def clean_review(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_directories():
    for path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_dataset():
    parquet_path = DATA_DIR / "user_reviews.parquet"
    csv_path = DATA_DIR / "user_reviews.csv"

    if not parquet_path.exists() and not csv_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Place user_reviews.parquet or user_reviews.csv in the data folder."
        )

    rows = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            required_columns = {"review_text", "score"}
            missing = required_columns - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV dataset is missing required columns: {sorted(missing)}")

            for row in reader:
                review = row.get("review_text")
                score = row.get("score")
                if review is None or score is None:
                    continue

                cleaned_review = clean_review(review)
                if not cleaned_review:
                    continue

                rows.append({
                    "cleaned_review": cleaned_review,
                    "sentiment": create_sentiment(float(score)),
                })

    else:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Parquet support is blocked by the Windows application control policy in this environment. "
                "Convert the data to CSV and save it as user_reviews.csv in the data folder."
            ) from exc

        table = pq.read_table(parquet_path)
        columns = set(table.column_names)
        required_columns = {"review_text", "score"}
        missing = required_columns - columns
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        for row in table.to_pylist():
            review = row.get("review_text")
            score = row.get("score")
            if review is None or score is None:
                continue

            cleaned_review = clean_review(review)
            if not cleaned_review:
                continue

            rows.append({
                "cleaned_review": cleaned_review,
                "sentiment": create_sentiment(float(score)),
            })

    return rows


def train_svm(X_train, y_train, X_test, y_test):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LinearSVC(class_weight="balanced", random_state=42)
    start = time.perf_counter()
    model.fit(X_train_vec, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)
    inference_latency_ms = (time.perf_counter() - start) * 1000 / len(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)

    joblib.dump(model, MODELS_DIR / "svm_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")

    return {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": matrix,
        "training_time": training_time,
        "inference_latency_ms": inference_latency_ms,
        "test_samples": len(y_test),
    }


def train_lstm(X_train, y_train, X_test, y_test, label_encoder):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(
        X_train_seq,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )
    X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(
        X_test_seq,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )

    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    class_counts = np.bincount(y_train_encoded, minlength=len(LABELS))
    class_weights = {
        index: len(y_train_encoded) / (len(LABELS) * count)
        for index, count in enumerate(class_counts)
        if count
    }

    tf.keras.utils.set_random_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=128),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(LABELS), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    start = time.perf_counter()
    history = model.fit(
        X_train_pad,
        y_train_encoded,
        validation_data=(X_test_pad, y_test_encoded),
        class_weight=class_weights,
        epochs=10,
        batch_size=32,
        verbose=0,
    )
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    test_predictions = model.predict(X_test_pad, verbose=0)
    inference_latency_ms = (time.perf_counter() - start) * 1000 / len(X_test)
    predicted_indices = np.argmax(test_predictions, axis=1)
    accuracy = accuracy_score(y_test_encoded, predicted_indices)
    macro_f1 = f1_score(y_test_encoded, predicted_indices, average="macro")
    weighted_f1 = f1_score(y_test_encoded, predicted_indices, average="weighted")

    report = classification_report(
        y_test_encoded,
        predicted_indices,
        labels=np.arange(len(LABELS)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test_encoded, predicted_indices, labels=np.arange(len(LABELS)))

    joblib.dump(tokenizer, MODELS_DIR / "tokenizer.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    model.save(MODELS_DIR / "lstm_model.keras")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": matrix,
        "history": history,
        "training_time": training_time,
        "inference_latency_ms": inference_latency_ms,
        "test_samples": len(y_test),
    }


def save_model_comparison(svm_results, lstm_results):
    csv_path = RESULTS_DIR / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "model",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "training_time_seconds",
            "inference_latency_ms_per_review",
        ])

        model_results = [("SVM", svm_results), ("LSTM", lstm_results)]
        evaluation = {"labels": LABELS, "models": {}}
        for model_name, results in model_results:
            report = results["classification_report"]
            macro = report["macro avg"]
            weighted = report["weighted avg"]
            writer.writerow([
                model_name,
                results["accuracy"],
                macro["precision"],
                macro["recall"],
                macro["f1-score"],
                weighted["precision"],
                weighted["recall"],
                weighted["f1-score"],
                results["training_time"],
                results["inference_latency_ms"],
            ])
            evaluation["models"][model_name] = {
                "accuracy": results["accuracy"],
                "training_time_seconds": results["training_time"],
                "inference_latency_ms_per_review": results["inference_latency_ms"],
                "test_samples": results["test_samples"],
                "classification_report": report,
                "confusion_matrix": results["confusion_matrix"].tolist(),
            }

    evaluation_path = RESULTS_DIR / "model_evaluation.json"
    with evaluation_path.open("w", encoding="utf-8") as json_file:
        json.dump(evaluation, json_file, indent=2)
        json_file.write("\n")


def main():
    ensure_directories()
    rows = load_dataset()

    X = [row["cleaned_review"] for row in rows]
    y = [row["sentiment"] for row in rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(LABELS)

    svm_results = train_svm(X_train, y_train, X_test, y_test)
    lstm_results = train_lstm(X_train, y_train, X_test, y_test, label_encoder)
    save_model_comparison(svm_results, lstm_results)

    print("Training complete.")
    print(f"SVM accuracy: {svm_results['accuracy']:.4f}")
    print(f"LSTM accuracy: {lstm_results['accuracy']:.4f}")

    gc.collect()


if __name__ == "__main__":
    main()
