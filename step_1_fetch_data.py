import nflreadpy as nfl
import polars as pl

def fetch_pbp_data(years: list[int]) -> pl.DataFrame:
    """
    Fetch and combine play-by-play data for given years using nflreadpy.
    """
    print(f"Fetching data for years: {years}")
    # nflreadpy concatenates automatically if you pass a list
    pbp = nfl.load_pbp(seasons=years)
    print(f"Loaded {len(pbp)} plays")
    return pbp

if __name__ == "__main__":
    # Example: 2000 to 2025
    years = list(range(2000, 2026))
    df = fetch_pbp_data(years)
    df.write_parquet("data/pbp_raw.parquet")  # Polars uses write_parquet