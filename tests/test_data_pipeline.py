import unittest

import pandas as pd

from src.data_loader import clean_match_data
from src.ml_model import _mirror_training_rows, make_feature_frame
from src.sentiment import create_sentiment, get_score_distribution
from src.review_analysis import analyze_review_context


def make_team_map(map_name: str, datetime: str, team1: str, team2: str) -> list[dict]:
    rows = []
    agents = ["Astra", "Breach", "Jett", "Sage", "Sova"]
    for team, opponent, score, opponent_score in (
        (team1, team2, "13", "8"),
        (team2, team1, "8", "13"),
    ):
        for index, agent in enumerate(agents):
            rows.append({
                "match-datetime": datetime,
                "patch": "6.0",
                "map": map_name,
                "team1": team1,
                "team2": team2,
                "team1-score": score if team == team1 else opponent_score,
                "team2-score": opponent_score if team == team1 else score,
                "player-name": f"{team}-player-{index}",
                "player-team": team,
                "agent": agent,
                "acs-t": "200",
                "acs-ct": "190",
                "kast": "0.7",
            })
    return rows


class SentimentLabelsTests(unittest.TestCase):
    def test_score_threshold_boundaries(self):
        self.assertEqual(create_sentiment(4), "Negative")
        self.assertEqual(create_sentiment(5), "Neutral")
        self.assertEqual(create_sentiment(6), "Neutral")
        self.assertEqual(create_sentiment(7), "Positive")

    def test_score_outside_supported_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            create_sentiment(-1)
        with self.assertRaises(ValueError):
            create_sentiment(11)

    def test_score_distribution_totals_and_percentages(self):
        result = get_score_distribution()
        self.assertEqual(sum(row["count"] for row in result["score_distribution"]), result["total_reviews"])
        categories = result["sentiment_distribution"].values()
        self.assertEqual(sum(item["count"] for item in categories), result["total_reviews"])
        self.assertAlmostEqual(sum(item["percentage"] for item in categories), 100.0, delta=0.02)


class ReviewAspectTests(unittest.TestCase):
    def test_requested_agent_and_map_examples(self):
        cases = (
            ("Jett is amazing and very fun.", {"Jett": "Positive"}),
            ("Raze feels weak after the patch.", {"Raze": "Negative"}),
            ("Sova is useful on Ascent.", {"Sova": "Positive"}),
            ("Jett is strong but Sage feels weak.", {"Jett": "Positive", "Sage": "Negative"}),
        )
        for review, expected in cases:
            with self.subTest(review=review):
                result = analyze_review_context(review)
                actual = {item["agent"]: item["sentiment"] for item in result["agents"]}
                self.assertEqual(actual, expected)
                if "Ascent" in review:
                    self.assertEqual([item["map"] for item in result["maps"]], ["Ascent"])
                if "patch" in review:
                    self.assertTrue(result["topics"]["Patch"]["detected"])

    def test_matchmaking_topic_does_not_create_an_agent(self):
        result = analyze_review_context("The matchmaking is terrible.")
        self.assertEqual(result["agents"], [])
        self.assertTrue(result["topics"]["Matchmaking"]["detected"])
        self.assertEqual(result["topics"]["Matchmaking"]["sentiment"], "Negative")

    def test_empty_review_returns_no_detected_entities(self):
        result = analyze_review_context("")
        self.assertEqual(result["agents"], [])
        self.assertEqual(result["maps"], [])

    def test_negation_flips_local_agent_polarity(self):
        result = analyze_review_context("Jett is not weak.")
        self.assertEqual(result["agents"][0]["sentiment"], "Positive")


class MatchCleaningTests(unittest.TestCase):
    def test_deduplicates_filters_bad_identity_and_checks_team_agent_lineups(self):
        rows = make_team_map("Ascent", "2023/01/01 12:00", "Team One", "Team Two")
        rows.extend(make_team_map("Bind", "2023/01/02 12:00", "Team Three", "Team Four"))
        rows[0]["acs-t"] = "-12"
        rows[0]["kast"] = "1.4"
        rows.append(rows[0].copy())
        rows[10]["agent"] = "1.42"

        cleaned, report = clean_match_data(pd.DataFrame(rows))

        self.assertEqual(report["raw_rows"], 21)
        self.assertEqual(report["invalid_identity_or_score_rows_removed"], 1)
        self.assertEqual(report["exact_duplicate_rows_removed"], 1)
        self.assertEqual(report["invalid_map_groups_removed"], 1)
        self.assertEqual(report["clean_map_groups"], 1)
        self.assertEqual(len(cleaned), 10)
        self.assertEqual(cleaned["map"].unique().tolist(), ["Ascent"])
        altered_row = cleaned[cleaned["player-name"].eq("Team One-player-0")].iloc[0]
        self.assertTrue(pd.isna(altered_row["acs-t"]))
        self.assertTrue(pd.isna(altered_row["kast"]))
        self.assertTrue(cleaned["player_won_map"].notna().all())

    def test_rejects_input_with_no_valid_complete_groups(self):
        rows = make_team_map("Ascent", "2023/01/01 12:00", "Team One", "Team Two")
        for row in rows:
            row["agent"] = "1.42"

        with self.assertRaisesRegex(ValueError, "No complete 5v5 map groups"):
            clean_match_data(pd.DataFrame(rows))


class MatchModelFeatureTests(unittest.TestCase):
    def test_mirrored_features_swap_team_lineups_and_labels(self):
        records = pd.DataFrame([{
            "map": "Ascent",
            "patch": "6.0",
            "team1_agents": ["Jett", "Sage"],
            "team2_agents": ["Astra", "Sova"],
            "team1_won": 1,
        }])
        agents = ["Astra", "Jett", "Sage", "Sova"]
        roles = ["Controller", "Duelist", "Initiator", "Sentinel"]
        role_by_agent = {
            "astra": "Controller",
            "jett": "Duelist",
            "sage": "Sentinel",
            "sova": "Initiator",
        }

        features = make_feature_frame(records, agents, roles, role_by_agent)
        mirrored_features, labels = _mirror_training_rows(records, features)

        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(mirrored_features.loc[0, "team1_agent_Jett"], 1)
        self.assertEqual(mirrored_features.loc[1, "team1_agent_Astra"], 1)
        self.assertEqual(mirrored_features.loc[0, "team2_role_Initiator"], 1)
        self.assertEqual(mirrored_features.loc[1, "team2_role_Duelist"], 1)


if __name__ == "__main__":
    unittest.main()