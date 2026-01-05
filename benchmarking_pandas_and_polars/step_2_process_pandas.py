import pandas as pd

def run_pandas(parquet_path: str):
    df = pd.read_parquet(parquet_path)

    fourth_downs = df[df["down"] == 4.0]

    attempts = fourth_downs[
        fourth_downs["play_type"].isin(["run", "pass"])
    ].copy()

    attempts["converted"] = attempts["fourth_down_converted"] == 1

    season_stats = (
        attempts.groupby("season")
        .agg(
            total_attempts=("converted", "count"),
            total_converted=("converted", "sum"),
            total_team_games=("game_id", "nunique"),
        )
        .reset_index()
    )

    season_stats["attempts_per_game"] = (
        season_stats["total_attempts"] / season_stats["total_team_games"]
    )
    season_stats["conversion_rate"] = (
        season_stats["total_converted"] / season_stats["total_attempts"]
    )

    return season_stats
