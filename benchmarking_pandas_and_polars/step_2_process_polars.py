import polars as pl

def run_polars(parquet_path: str):
    df = pl.read_parquet(parquet_path)

    fourth_downs = df.filter(pl.col("down") == 4.0)

    attempts = fourth_downs.filter(
        pl.col("play_type").is_in(["run", "pass"])
    ).with_columns(
        (pl.col("fourth_down_converted") == 1).alias("converted")
    )

    season_stats = (
        attempts.group_by("season")
        .agg([
            pl.count().alias("total_attempts"),
            pl.sum("converted").alias("total_converted"),
            pl.n_unique("game_id").alias("total_team_games"),
        ])
        .with_columns([
            (pl.col("total_attempts") / pl.col("total_team_games")).alias("attempts_per_game"),
            (pl.col("total_converted") / pl.col("total_attempts")).alias("conversion_rate"),
        ])
    )

    return season_stats
