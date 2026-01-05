import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

def load_pbp() -> pl.DataFrame:
    return pl.read_parquet("data/pbp_raw.parquet")

def prepare_data(df: pl.DataFrame, min_ydstogo=1, max_ydstogo=10, min_situations=30):
    # Same as before, but keep 'total' for sizing
    fourth_downs = df.filter(
        (pl.col('down') == 4.0) &
        (pl.col('ydstogo').is_between(min_ydstogo, max_ydstogo)) &
        (pl.col('season') >= 2000) &
        (pl.col('yardline_100').is_not_null())
    )
    
    goes = fourth_downs.filter(pl.col('play_type').is_in(['pass', 'run']))
    
    fourth_downs = fourth_downs.with_columns(
        pl.when(pl.col('season') < 2010).then(pl.lit('2000–2009'))
          .when(pl.col('season') < 2015).then(pl.lit('2010–2014'))
          .when(pl.col('season') < 2020).then(pl.lit('2015–2019'))
          .otherwise(pl.lit('2020–2025')).alias('era'),
        (100 - pl.col('yardline_100')).alias('field_pos')
    )
    goes = goes.with_columns(
        pl.when(pl.col('season') < 2010).then(pl.lit('2000–2009'))
          .when(pl.col('season') < 2015).then(pl.lit('2010–2014'))
          .when(pl.col('season') < 2020).then(pl.lit('2015–2019'))
          .otherwise(pl.lit('2020–2025')).alias('era'),
        (100 - pl.col('yardline_100')).alias('field_pos')
    )
    
    # Era aggregation (keep total)
    total_era = fourth_downs.group_by(['era', 'field_pos']).agg(pl.count().alias('total'))
    go_era = goes.group_by(['era', 'field_pos']).agg(pl.count().alias('goes'))
    era_df = total_era.join(go_era, on=['era', 'field_pos'], how='left')
    era_df = era_df.with_columns(
        (pl.col('goes') / pl.col('total')).fill_null(0).alias('go_rate')
    ).filter(pl.col('total') >= min_situations)
    
    # Season aggregation (keep total)
    total_season = fourth_downs.group_by(['season', 'field_pos']).agg(pl.count().alias('total'))
    go_season = goes.group_by(['season', 'field_pos']).agg(pl.count().alias('goes'))
    season_df = total_season.join(go_season, on=['season', 'field_pos'], how='left')
    season_df = season_df.with_columns(
        (pl.col('goes') / pl.col('total')).fill_null(0).alias('go_rate')
    ).filter(pl.col('total') >= min_situations)
    
    return era_df.to_pandas(), season_df.to_pandas()

def add_football_field(ax):
    ax.set_facecolor('#2E7D32')
    for y in range(0, 101, 5):
        lw = 3 if y % 10 == 0 else 1
        alpha = 1 if y % 10 == 0 else 0.6
        ax.axvline(y, ymin=0, ymax=0.95, color='white', linewidth=lw, alpha=alpha, zorder=1)
    ax.axvline(0, color='white', linewidth=4, zorder=1)
    ax.axvline(100, color='white', linewidth=4, zorder=1)
    for y in [10, 20, 30, 40]:
        ax.text(y, 1.05, str(y), color='white', ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(100 - y, 1.05, str(y), color='white', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(50, 1.05, '50', color='white', ha='center', va='center', fontsize=18, fontweight='bold')
    ax.set_ylim(0, 1.15)

if __name__ == "__main__":
    df = load_pbp()
    era_df, season_df = prepare_data(df)
    
    # Scale sizes (log for better spread, then linear map)
    era_df['size'] = era_df['total'] ** 0.5 * 10  # Bigger where more data
    season_df['size'] = season_df['total'] ** 0.5 * 5
    
    sns.set(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 1, figsize=(16, 13), sharex=True)
    
    era_palette = {
        '2000–2009': 'navy',
        '2010–2014': 'royalblue',
        '2015–2019': 'darkorange',
        '2020–2025': 'crimson'
    }
    
    # Top: Era (larger points)
    sns.scatterplot(
        data=era_df,
        x='field_pos',
        y='go_rate',
        hue='era',
        hue_order=['2000–2009', '2010–2014', '2015–2019', '2020–2025'],
        palette=era_palette,
        size='size',
        sizes=(50, 400),  # Range for visibility
        edgecolor='black',
        linewidth=1,
        alpha=0.9,
        ax=axes[0],
        legend=False,
        zorder=10
    )
    add_football_field(axes[0])
    axes[0].set_title('Fourth Down Go-for-It Rate by Field Position — Grouped by Era\n(4th & 1–10 yards to go | Point size = # of situations)', fontsize=16, pad=20)
    axes[0].set_ylabel('Go-for-It %', fontsize=14)
    axes[0].set_xlabel('')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[0].legend(title='Era\n(Navy = oldest → Crimson = newest/most aggressive)', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    
    # Bottom: Season
    sns.scatterplot(
        data=season_df,
        x='field_pos',
        y='go_rate',
        hue='season',
        palette='rocket',
        size='size',
        sizes=(20, 200),
        edgecolor='black',
        linewidth=0.8,
        alpha=0.85,
        ax=axes[1],
        legend=False,
        zorder=10
    )
    add_football_field(axes[1])
    axes[1].set_title('Fourth Down Go-for-It Rate by Field Position — Year by Year\n(4th & 1–10 yards to go | Point size = # of situations)', fontsize=16, pad=20)
    axes[1].set_ylabel('Go-for-It %', fontsize=14)
    axes[1].set_xlabel('Field Position (Left = Deep Own Territory → Right = Near Opponent Goal)', fontsize=14)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[1].legend(title='Season\n(Darker purple = earlier → Lighter = recent/more aggressive)', loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=2, frameon=True)
    
    axes[1].set_xlim(0, 100)
    
    fig.text(0.5, 0.02,
             'Left side = own territory (historically low rates) | Right side = opponent territory (higher rates)\n'
             'Recent points cluster higher & farther left = increased 4th-down aggression over time',
             ha='center', fontsize=12, linespacing=1.5)
    
    plt.tight_layout(rect=[0, 0.05, 0.82, 1])
    plt.show()