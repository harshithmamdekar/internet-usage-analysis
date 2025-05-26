import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA

# Load Data
df = pd.read_csv('C:/Users/hp/Downloads/World-Internet-Access-Data-main/World-Internet-Access-Data-main/internet_users.csv')

# Quick data overview
print(df.head())
print(df.isnull().sum())

# Convert share to percentage
df['share_percent'] = df['share'] * 100

# --- 1. Global Internet Usage Trend ---
global_trend = df.groupby("year")["users"].mean()
plt.figure(figsize=(10,5))
plt.plot(global_trend.index, global_trend.values)
plt.title("Global Internet Usage Over Time")
plt.xlabel("Year")
plt.ylabel("Average % Population Online")
plt.show()

# --- 2. Interactive Plot for Selected Countries ---
countries = ['India', 'United States', 'China']
fig = px.line(df[df['entity'].isin(countries)], x='year', y='users', color='entity',
              title='Interactive Internet Usage Over Time')
fig.show()

# --- 3. Clustering Countries by Usage Pattern ---

# Pivot data: entities as rows, years as columns
pivot_df = df.pivot(index='entity', columns='year', values='users').fillna(0)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
pivot_df['Cluster'] = kmeans.fit_predict(pivot_df)

# Visualize clusters using 2010 vs 2020 penetration
plt.figure(figsize=(10,6))
sns.scatterplot(data=pivot_df, x=2010, y=2020, hue='Cluster', palette='tab10')
plt.title('Clusters of Countries by Internet Penetration in 2010 vs 2020')
plt.xlabel('Internet Users in 2010 (%)')
plt.ylabel('Internet Users in 2020 (%)')
plt.legend(title='Cluster')
plt.show()

# --- 4. CAGR Calculation ---

def calculate_cagr(start, end, periods):
    if start == 0 or end == 0:
        return 0
    return ((end / start) ** (1/periods) - 1) * 100

cagr_list = []
for entity in df['entity'].unique():
    temp = df[df['entity'] == entity].sort_values('year')
    if temp.shape[0] > 1:
        start_users = temp['users'].iloc[0]
        end_users = temp['users'].iloc[-1]
        periods = temp['year'].iloc[-1] - temp['year'].iloc[0]
        cagr = calculate_cagr(start_users, end_users, periods) if periods > 0 else 0
        cagr_list.append({'entity': entity, 'CAGR': cagr})

cagr_df = pd.DataFrame(cagr_list).sort_values(by='CAGR', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(data=cagr_df.head(10), x='CAGR', y='entity', palette='viridis')
plt.title("Top 10 Countries by Internet User CAGR")
plt.xlabel("Compound Annual Growth Rate (%)")
plt.ylabel("Country")
plt.show()

# --- 5. ARIMA Forecasting for India ---

india_ts = df[df['entity']=='India'].set_index('year')['users'].sort_index()

# Fit ARIMA (order can be tuned)
model = ARIMA(india_ts, order=(1,1,1))
model_fit = model.fit()

forecast = model_fit.get_forecast(steps=10)
forecast_df = forecast.summary_frame()

plt.figure(figsize=(10,5))
plt.plot(india_ts.index, india_ts, label='Historical', marker='o')
plt.plot(range(2021, 2031), forecast_df['mean'], label='Forecast', marker='o', color='red')
plt.fill_between(range(2021, 2031), forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='pink', alpha=0.3)
plt.title('ARIMA Forecast for Internet Usage in India')
plt.xlabel('Year')
plt.ylabel('Internet Users (%)')
plt.legend()
plt.show()

# --- 6. Regional Internet Penetration Heatmap ---

# Check if 'region' column exists, else skip
if 'entity' in df.columns:
    region_year = df.groupby(['year', 'entity'])['users'].mean().reset_index()
    heatmap_data = region_year.pivot(index='entity', columns='year', values='users')

    plt.figure(figsize=(12,7))
    sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=0.5)
    plt.title('Regional Internet Penetration Over Time')
    plt.xlabel('Year')
    plt.ylabel('Region')
    plt.show()
else:
    print("Region column not found in data; skipping regional heatmap.")
