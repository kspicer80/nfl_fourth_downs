import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

def load_pbp() -> pl.DataFrame:
    return pl.read_parquet("data/pbp_raw.parquet")

def prepare_heatmap_data(df: pl.DataFrame, min_ydstogo: int = 1, max_ydstogo: int = 10) -> pl.DataFrame:
    # Filter to fourth downs (exclude kneels, spikes, etc.)
    fourth_downs = df.filter(
        (pl.col('down') == 4.0) &
        (pl.col('ydstogo').is_between(min_ydstogo, max_ydstogo)) &
        (pl.col('season') >= 2000)
    )
    
    # Define "go for it"
    goes = fourth_downs.filter(pl.col('play_type').is_in(['pass', 'run']))
    
    # Bin yardline and season
    heatmap = fourth_downs.with_columns([
        pl.col('yardline_100').floordiv(5).mul(5).alias('yardline_bin'),
        # Optional: group into eras for denser data
        pl.when(pl.col('season') < 2010).then(pl.lit('2000-2009'))
          .when(pl.col('season') < 2015).then(pl.lit('2010-2014'))
          .when(pl.col('season') < 2020).then(pl.lit('2015-2019'))
          .otherwise(pl.lit('2020-2025')).alias('era')
    ])
    
    total_fourth = heatmap.group_by(['era', 'yardline_bin']).agg(pl.count().alias('total'))
    go_fourth = goes.with_columns([
        pl.col('yardline_100').floordiv(5).mul(5).alias('yardline_bin'),
        pl.when(pl.col('season') < 2010).then(pl.lit('2000-2009'))
          .when(pl.col('season') < 2015).then(pl.lit('2010-2014'))
          .when(pl.col('season') < 2020).then(pl.lit('2015-2019'))
          .otherwise(pl.lit('2020-2025')).alias('era')
    ]).group_by(['era', 'yardline_bin']).agg(pl.count().alias('goes'))
    
    heatmap_data = total_fourth.join(go_fourth, on=['era', 'yardline_bin'], how='left')
    heatmap_data = heatmap_data.with_columns(
        (pl.col('goes') / pl.col('total')).fill_null(0).alias('go_rate')
    ).filter(pl.col('total') >= 20)  # Minimum situations for reliability
    
    return heatmap_data.to_pandas()  # Convert to pandas for seaborn

if __name__ == "__main__":
    df = load_pbp()
    hm_df = prepare_heatmap_data(df)
    
    # Pivot for heatmap
    pivot = hm_df.pivot(index='era', columns='yardline_bin', values='go_rate')
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='YlOrRd', linewidths=.5, cbar_kws={'label': 'Go-for-It %'})
    plt.title('NFL Fourth Down Go-for-It Rate by Field Position and Era\n(1-10 yards to go, min 20 situations per bin)')
    plt.xlabel('Yards to Opponent End Zone (yardline_100 binned)')
    plt.ylabel('Season Era')
    plt.gca().invert_yaxis()  # Newest era on top
    plt.show()