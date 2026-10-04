# Dataset Provenance

- Title: Valorant Metacritic Reviews
- Creator credit: Larxel (`andrewmvd`) and one collaborator
- Source: [Kaggle dataset](https://www.kaggle.com/datasets/andrewmvd/valorant-metacritic-reviews)
- DOI: <https://doi.org/10.34740/kaggle/dsv/1236634>
- License: Original Authors, CC BY-NC-SA 4.0
- Snapshot: Metacritic reviews collected through June 11, 2020
- File: `user_reviews.parquet` (1,009 rows, 7 columns)

This project cleans `review_text` and derives the three sentiment labels from
`score`; those are adaptations of the source data. Credit the original authors
and preserve the license terms for any redistributed adapted dataset. The
license is non-commercial.

## Professional Match Data


This source has no event, region, or explicit match ID. Match-map groups must be
reconstructed from date, patch, map, teams, and scores. The T/CT columns contain
player metrics; they are not side win rates. Some raw records have invalid agent
or score fields and duplicate map-player rows. The preprocessing report records
the exact exclusions and metric sanitization. The source snapshot is historical
and does not cover maps or agents released after April 2023. The creator states
that records were scraped from VLR.gg; the CC0 listing applies to the Kaggle
dataset, and no VLR data license is asserted by this project.

## Agent Role Metadata

- Source: [Valorant-API](https://valorant-api.com/v1/agents?isPlayableCharacter=true)
- Retrieved: September 30, 2026
- File: `agents/agent_roles.csv`
- Fields retained: canonical agent name and role only

Valorant-API is a community-maintained, unofficial mirror of game metadata. It
is used only to attach role labels to agent names; all performance values come
from the match dataset. The historical source spelling `Kayo` is normalized to
the catalog name `KAY/O` for joins.