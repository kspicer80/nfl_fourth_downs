import polars as pl

def load_pbp() -> pl.DataFrame:
    return pl.read_parquet("data/pbp_raw.parquet")

def filter_fourth_down_attempts(df: pl.DataFrame) -> pl.DataFrame:
    fourth_downs = df.filter(pl.col('down') == 4.0)
    
    attempts = fourth_downs.filter(pl.col('play_type').is_in(['pass', 'run']))
    
    attempts = attempts.with_columns(
        (pl.col('fourth_down_converted') == 1).alias('converted')
    )
    
    return attempts

def aggregate_season_attempts(attempts: pl.DataFrame) -> pl.DataFrame:
    # Unique games per season (each game appears once, but two teams play)
    games_per_season = attempts.group_by('season').agg(pl.col('game_id').n_unique() * 2)
    
    season_stats = attempts.group_by('season').agg(
        total_attempts = pl.col('play_id').count(),
        total_converted = pl.col('converted').sum(),
        total_team_games = pl.col('game_id').n_unique() * 2  # each game = 2 team-games
    ).join(games_per_season, on='season', how='left')
    
    season_stats = season_stats.with_columns(
        (pl.col('total_attempts') / pl.col('total_team_games')).alias('attempts_per_game'),
        (pl.col('total_converted') / pl.col('total_attempts')).alias('conversion_rate')
    )
    
    return season_stats

if __name__ == "__main__":
    df = load_pbp()
    attempts = filter_fourth_down_attempts(df)
    season_trends = aggregate_season_attempts(attempts)
    season_trends.write_csv("data/season_fourth_down_trends.csv")
    print(season_trends)