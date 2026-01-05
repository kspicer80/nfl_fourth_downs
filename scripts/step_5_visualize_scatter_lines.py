import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

def load_pbp() -> pl.DataFrame:
    return pl.read_parquet("data/pbp_raw.parquet")

def prepare_scatter_data(df: pl.DataFrame, min_ydstogo=1, max_ydstogo=10, min_situations=30) -> pl.DataFrame:
    # Filter relevant fourth downs
    fourth_downs = df.filter(
        (pl.col('down') == 4.0) &
        (pl.col('ydstogo').is_between(min_ydstogo, max_ydstogo)) &
        (pl.col('season') >= 2000) &
        (pl.col('yardline_100').is_not_null())
    )
    
    # Go for it attempts
    goes = fourth_downs.filter(pl.col('play_type').is_in(['pass', 'run']))
    
    # Group into eras (adjust as desired)
    data = fourth_downs.with_columns(
        pl.when(pl.col('season') < 2010).then(pl.lit('2000–2009'))
          .when(pl.col('season') < 2015).then(pl.lit('2010–2014'))
          .when(pl.col('season') < 2020).then(pl.lit('2015–2019'))
          .otherwise(pl.lit('2020–2025')).alias('era')
    )
    
    goes = goes.with_columns(
        pl.when(pl.col('season') < 2010).then(pl.lit('2000–2009'))
          .when(pl.col('season') < 2015).then(pl.lit('2010–2014'))
          .when(pl.col('season') < 2020).then(pl.lit('2015–2019'))
          .otherwise(pl.lit('2020–2025')).alias('era')
    )
    
    # Aggregate by era and yardline_100
    total = data.group_by(['era', 'yardline_100']).agg(pl.count().alias('total'))
    go = goes.group_by(['era', 'yardline_100']).agg(pl.count().alias('goes'))
    
    agg = total.join(go, on=['era', 'yardline_100'], how='left')
    agg = agg.with_columns(
        (pl.col('goes') / pl.col('total')).fill_null(0).alias('go_rate')
    ).filter(pl.col('total') >= min_situations)  # Reliability filter
    
    return agg.to_pandas()

if __name__ == "__main__":
    df = load_pbp()
    scatter_df = prepare_scatter_data(df)
    
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Use lineplot with markers for clarity
    sns.lineplot(
        data=scatter_df,
        x='yardline_100',
        y='go_rate',
        hue='era',
        marker='o',
        linewidth=3,
        markersize=6,
        palette='viridis'
    )
    
    plt.title('NFL Fourth Down Go-for-It Rate by Field Position and Era\n(4th & 1–10, min 30 situations per yard line)')
    plt.xlabel('Yards to Opponent End Zone (yardline_100)\n99 = own 1-yard line, 1 = opponent goal line')
    plt.ylabel('Go-for-It Percentage')
    plt.gca().invert_xaxis()  # Makes it read left-to-right: own territory → opponent territory
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add a vertical line at midfield for reference
    plt.axvline(x=50, color='gray', linestyle='--', alpha=0.7, label='Midfield')
    
    plt.legend(title='Era', loc='upper left')
    plt.tight_layout()
    plt.show()