import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_csv("data/season_fourth_down_trends.csv")

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='season', y='attempts_per_game', marker='o')
plt.title('NFL Fourth Down Conversion Attempts per Team per Game (League Average)')
plt.ylabel('Attempts per Game')
plt.xlabel('Season')
plt.show()

# Bonus: conversion rate over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='season', y='conversion_rate', marker='o', color='green')
plt.title('Fourth Down Conversion Success Rate Over Time')
plt.ylabel('Success Rate')
plt.ylim(0.4, 0.7)
plt.show()