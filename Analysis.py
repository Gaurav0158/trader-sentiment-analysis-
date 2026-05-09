import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
trades = pd.read_csv('historical_data.csv')
sentiment = pd.read_csv('fear_greed.csv')
print(trades.head())
print(trades.shape)
print(trades.info())
print(sentiment.head())
print(sentiment.shape)
print(trades.isnull().sum())
print(sentiment.isnull().sum())
print(trades.duplicated().sum())
print(sentiment.duplicated().sum())
trades = trades.drop_duplicates()
sentiment = sentiment.drop_duplicates()
trades['Timestamp IST'] = pd.to_datetime(
    trades['Timestamp IST'],
    dayfirst=True
)

trades['date'] = trades['Timestamp IST'].dt.date
sentiment['date'] = pd.to_datetime(sentiment['date']).dt.date
merged_df = pd.merge(trades, sentiment, on='date', how='left')
print(merged_df.head())
print(merged_df.shape)

merged_df['is_profit'] = np.where(merged_df['Closed PnL'] > 0, 1, 0)

merged_df['trade_size_category'] = pd.qcut(
    merged_df['Size USD'],
    q=2,
    labels=['Low Size', 'High Size']
)
merged_df['Direction'] = merged_df['Side'].str.upper()

daily_pnl = merged_df.groupby('date')['Closed PnL'].sum().reset_index()
print(daily_pnl.head())
win_rate = merged_df['is_profit'].mean() * 100
print(f'Overall Win Rate: {win_rate:.2f}%')
avg_trade_size = merged_df['Size USD'].mean()
print(avg_trade_size)
trades_per_day = merged_df.groupby('date').size().reset_index(name='trade_count')
print(trades_per_day.head())
long_short = merged_df['Direction'].value_counts(normalize=True) * 100
print(long_short)
pnl_by_sentiment = merged_df.groupby('classification')['Closed PnL'].mean().reset_index()
print(pnl_by_sentiment)

plt.figure(figsize=(8,5))
sns.barplot(data=pnl_by_sentiment, x='classification', y='Closed PnL')
plt.title('Average PnL by Sentiment')
plt.show()

win_by_sentiment = merged_df.groupby('classification')['is_profit'].mean().reset_index()
win_by_sentiment['is_profit'] *= 100

print(win_by_sentiment)

plt.figure(figsize=(8,5))
sns.barplot(data=win_by_sentiment, x='classification', y='is_profit')
plt.title('Win Rate by Sentiment')
plt.ylabel('Win Rate %')
plt.show()
trade_freq = merged_df.groupby('classification').size().reset_index(name='trade_count')
print(trade_freq)

plt.figure(figsize=(8,5))
sns.barplot(data=trade_freq, x='classification', y='trade_count')
plt.title('Trade Frequency by Sentiment')
plt.show()

size_analysis = merged_df.groupby('classification')['Size USD'].mean().reset_index()
print(size_analysis)

plt.figure(figsize=(8,5))
sns.barplot(data=size_analysis, x='classification', y='Size USD')
plt.title('Average Trade Size by Sentiment')
plt.show()

bias = pd.crosstab(
    merged_df['classification'],
    merged_df['Direction'],
    normalize='index'
) * 100

print(bias)
bias.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title('Long vs Short Bias by Sentiment')
plt.ylabel('Percentage')
plt.show()

trader_frequency = merged_df.groupby('Account').size().reset_index(name='trade_count')

median_trade_count = trader_frequency['trade_count'].median()

trader_frequency['segment'] = np.where(
    trader_frequency['trade_count'] >= median_trade_count,
    'Frequent Trader',
    'Infrequent Trader'
)

print(trader_frequency.head())
merged_df = pd.merge(
    merged_df,
    trader_frequency[['Account', 'segment']],
    on='Account',
    how='left'
)
segment_perf = merged_df.groupby('segment')['Closed PnL'].mean().reset_index()
print(segment_perf)

plt.figure(figsize=(8,5))
sns.barplot(data=segment_perf, x='segment', y='Closed PnL')
plt.title('PnL by Trader Segment')
plt.show()
size_perf = merged_df.groupby('trade_size_category')['Closed PnL'].mean().reset_index()
print(size_perf)

plt.figure(figsize=(8,5))
sns.barplot(data=size_perf, x='trade_size_category', y='Closed PnL')
plt.title('PnL by Trade Size Category')
plt.show()

daily_pnl_series = merged_df.groupby('date')['Closed PnL'].sum()
rolling_volatility = daily_pnl_series.rolling(window=7).std()

plt.figure(figsize=(12,6))
rolling_volatility.plot()
plt.title('Rolling PnL Volatility (Drawdown Proxy)')
plt.show()
