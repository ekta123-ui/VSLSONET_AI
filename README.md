# VALOSENT AI

VALOSENT AI is an MCA research application combining VALORANT review sentiment analysis with historical professional-match analytics. It uses real source data; it does not generate synthetic match statistics.

## Run Locally

Use the project Python environment and install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Prepare the match analytics dataset and model:

```powershell
python -m src.data_loader
python -m src.ml_model
```

Train/retrain the sentiment models, then start Flask:

```powershell
python train_models.py
python app.py
```

The app serves at `http://127.0.0.1:5000` by default. Match data is kept under `data/raw/` and processed output under `data/processed/`; these local data files are ignored by Git. Follow the provenance notes in [data/README.md](data/README.md) when acquiring or redistributing source data.

## Deploy on Render

The included `render.yaml` configures a Flask web service. Push the project to GitHub, create a Render Blueprint from that repository, and apply the `valosent-ai` service. Render installs `requirements.txt`, runs the app with Gunicorn, and checks `/health`. Ensure any required data and trained model files are included in the deployment; they are not generated automatically during deploy.

Run the focused regression suite with:

```powershell
python -m unittest discover -s tests
```

## Implemented Features

- Dashboard coverage summaries generated from the cleaned match dataset.
- Dynamic map and agent catalogs; role metadata is maintained separately from performance data.
- Agent/map pick rate, map win rate, and T/CT median ACS with sample counts.
- Map/role/side recommendations with explicit support thresholds and a Wilson lower-bound score for overall map win rate.
- Historical compositions and pairings only when repeated source records exist.
- Review sentiment prediction using LinearSVC and LSTM, plus saved model evaluations.
- A match-outcome Logistic Regression experiment compared with a majority baseline.

## Important Limitations

The review labels are derived from Metacritic scores, not human sentiment annotations. The match snapshot ends in April 2023 and lacks event, region, explicit match IDs, and starting-side data. T/CT fields are player performance metrics, not side win rates. The current match classifier has lower accuracy than the majority baseline and is presented as an experiment, not a live recommendation signal.

Review-level SVM/LSTM predictions use the score-derived training labels. Agent and topic polarity shown for a submitted review is a separate, transparent local-context word-list heuristic; it is not a trained or human-validated aspect model. Sentiment-by-agent counts include only reviews with a detected agent name and are small for this June 2020 review snapshot. Common words that overlap agent or map names may still be ambiguous. Do not interpret these mention-based estimates as ground-truth agent sentiment.

The review dataset is CC BY-NC-SA 4.0 and non-commercial. The match dataset is listed as CC0 on Kaggle and is described by its creator as scraped from VLR.gg; this project does not assert a VLR license. See [data/README.md](data/README.md) for provenance.