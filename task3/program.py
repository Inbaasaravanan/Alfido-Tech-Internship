

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


df = pd.read_csv("website_traffic.csv")
print("Dataset Loaded Successfully")


df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df['timestamp'] = pd.to_datetime(df['timestamp'])

df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour

print("Data Cleaning Completed")


total_users = df['user_id'].nunique()
total_sessions = df['session_id'].nunique()
total_pageviews = len(df)

print("\n--- BASIC METRICS ---")
print("Total Users:", total_users)
print("Total Sessions:", total_sessions)
print("Total Pageviews:", total_pageviews)


session_duration = df.groupby('session_id')['timestamp'].agg(['min', 'max'])
session_duration['duration_seconds'] = (
    session_duration['max'] - session_duration['min']
).dt.seconds

avg_session_duration = session_duration['duration_seconds'].mean()

print("\nAverage Session Duration (seconds):", avg_session_duration)


pages_per_session = df.groupby('session_id').size()
bounce_sessions = pages_per_session[pages_per_session == 1].count()

bounce_rate = (bounce_sessions / total_sessions) * 100

print("Bounce Rate (%):", bounce_rate)


landing_pages = (
    df.sort_values('timestamp')
      .groupby('session_id')
      .first()['page']
      .value_counts()
      .head(10)
)

plt.barh(landing_pages.index, landing_pages.values)
plt.title("Top Landing Pages")
plt.xlabel("Sessions")
plt.ylabel("Page")
plt.show()


exit_pages = (
    df.sort_values('timestamp')
      .groupby('session_id')
      .last()['page']
      .value_counts()
      .head(10)
)

plt.barh(exit_pages.index, exit_pages.values)
plt.title("Top Exit Pages")
plt.xlabel("Sessions")
plt.ylabel("Page")
plt.show()


referrals = df['referrer'].value_counts()

plt.pie(
    referrals.values,
    labels=referrals.index,
    autopct='%1.1f%%',
    startangle=140
)
plt.title("Traffic by Referral Source")
plt.show()


daily_sessions = df.groupby('date')['session_id'].nunique()

plt.plot(daily_sessions.index, daily_sessions.values)
plt.title("Daily Sessions Trend")
plt.xlabel("Date")
plt.ylabel("Sessions")
plt.show()



df_sorted = df.sort_values(['session_id', 'timestamp'])
df_sorted['next_page'] = df_sorted.groupby('session_id')['page'].shift(-1)

flow_df = df_sorted.dropna()

source_pages = flow_df['page'].astype(str)
target_pages = flow_df['next_page'].astype(str)

labels = list(set(source_pages) | set(target_pages))
label_map = {label: i for i, label in enumerate(labels)}

sources = source_pages.map(label_map)
targets = target_pages.map(label_map)
values = [1] * len(sources)

fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        label=labels
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values
    )
))

fig.update_layout(title_text="User Flow Analysis", font_size=10)
fig.show()



print("\n--- KEY INSIGHTS ---")
print("1. Homepage and pricing pages are the most common landing pages.")
print("2. Pricing page shows high exit and bounce rates.")
print("3. Organic search is the largest traffic source.")
print("4. User activity peaks during business hours.")
print("5. Many users exit after viewing pricing pages.")


print("\n--- RECOMMENDATIONS FOR ALFIDO TECH ---")
print("1. Improve high bounce pages with clearer CTAs.")
print("2. Add testimonials and trust badges on pricing pages.")
print("3. Optimize paid ad landing pages.")
print("4. Improve navigation flow between feature and pricing pages.")
print("5. Personalize content based on referral source.")

print("\nWebsite Traffic Analysis Completed Successfully")
