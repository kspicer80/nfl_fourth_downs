import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

def load_pbp() -> pl.DataFrame:
    return pl.read_parquet("data/pbp_raw.parquet")

def prepare_data(df: pl.DataFrame, min_ydstogo=1, max_ydstogo=10, min_situations=30):
    fourth_downs = df.filter(
        (pl.col('down') == 4.0) &
        (pl.col('ydstogo').is_between(min_ydstogo, max_ydstogo)) &
        (pl.col('season') >= 2000) &
        (pl.col('yardline_100').is_not_null())
    )
    
    goes = fourth_downs.filter(pl.col('play_type').is_in(['pass', 'run']))
    
    # Add era
    fourth_downs = fourth_downs.with_columns(
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
    
    # Aggregate by era + yardline
    total_era = fourth_downs.group_by(['era', 'yardline_100']).agg(pl.count().alias('total'))
    go_era = goes.group_by(['era', 'yardline_100']).agg(pl.count().alias('goes'))
    era_df = total_era.join(go_era, on=['era', 'yardline_100'], how='left')
    era_df = era_df.with_columns(
        (pl.col('goes') / pl.col('total')).fill_null(0).alias('go_rate')
    ).filter(pl.col('total') >= min_situations)
    
    # Aggregate by season + yardline
    total_season = fourth_downs.group_by(['season', 'yardline_100']).agg(pl.count().alias('total'))
    go_season = goes.group_by(['season', 'yardline_100']).agg(pl.count().alias('goes'))
    season_df = total_season.join(go_season, on=['season', 'yardline_100'], how='left')
    season_df = season_df.with_columns(
        (pl.col('goes') / pl.col('total')).fill_null(0).alias('go_rate')
    ).filter(pl.col('total') >= min_situations)
    
    return era_df.to_pandas(), season_df.to_pandas()

if __name__ == "__main__":
    df = load_pbp()
    era_df, season_df = prepare_data(df)
    
    sns.set(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # === Top: By Era ===
    sns.lineplot(
        data=era_df,
        x='yardline_100',
        y='go_rate',
        hue='era',
        marker='o',
        linewidth=3.5,
        markersize=7,
        palette='viridis',
        ax=axes[0]
    )
    axes[0].set_title('Fourth Down Go-for-It Rate by Field Position — Grouped by Era\n(4th & 1–10 yards to go)')
    axes[0].set_ylabel('Go-for-It %')
    axes[0].invert_xaxis()
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[0].axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    axes[0].legend(title='Era', loc='upper left')
    
    # === Bottom: By Individual Season ===
    sns.lineplot(
        data=season_df,
        x='yardline_100',
        y='go_rate',
        hue='season',
        palette='crest',  # Beautiful sequential gradient
        linewidth=2,
        alpha=0.9,
        ax=axes[1]
    )
    axes[1].set_title('Fourth Down Go-for-It Rate by Field Position — Year by Year\n(4th & 1–10 yards to go)')
    axes[1].set_xlabel('Yards to Opponent End Zone (yardline_100)\n99 = own 1-yard line → 1 = opponent goal line')
    axes[1].set_ylabel('Go-for-It %')
    axes[1].invert_xaxis()
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[1].axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    axes[1].legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()