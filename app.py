
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(layout="wide", page_title="Fashion AI Control Tower - Demo")

DATA_PATH = Path(__file__).parent / "dummy_data.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df

df = load_data()

st.title("Fashion AI Control Tower — Demo")
st.markdown("One-stop prototype: demand forecasting, inventory digital twin, returns hub, segmentation & sustainability signals.")

# Top controls
col1, col2, col3 = st.columns([2,2,1])
with col1:
    region = st.selectbox("Region / Warehouse", options=["Dubai","Riyadh","Mumbai","HoChiMinh"])
with col2:
    sku_choice = st.selectbox("Choose SKU (for drilldown)", options=sorted(df["sku"].unique()))
with col3:
    days_ahead = st.slider("Forecast horizon (days)", 7, 30, 14)

# Basic KPI panel
recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=30)]
kpi1 = int(recent["demand"].mean())
kpi2 = int(recent["demand"].std())
kpi3 = int(recent["promo"].sum())
k1, k2, k3 = st.columns(3)
k1.metric("Avg daily demand (30d)", kpi1)
k2.metric("Demand volatility (std)", kpi2)
k3.metric("Promos last 30d", kpi3)

st.markdown("## Demand Forecasting (Model demo)")
# Train a simple RandomForest per-SKU using basic features (lag features)
def prepare_features(df_sku):
    df_sku = df_sku.sort_values("date").copy()
    df_sku["lag1"] = df_sku["demand"].shift(1).fillna(method="bfill")
    df_sku["lag7"] = df_sku["demand"].shift(7).fillna(method="bfill")
    X = df_sku[["lag1","lag7","social_buzz","promo","weather_index"]]
    y = df_sku["demand"]
    return X, y, df_sku

df_sku = df[df["sku"]==sku_choice].copy()
X, y, df_sku_prep = prepare_features(df_sku)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
# Build inputs for forecasting next days (naive: last observed values with slight noise)
last_row = df_sku_prep.iloc[-1]
preds = []
input_row = last_row.copy()
for i in range(days_ahead):
    inp = np.array([[input_row["demand"], input_row["demand"], input_row["social_buzz"], input_row["promo"], input_row["weather_index"]]])
    p = model.predict(inp)[0]
    preds.append(max(0, p))
    # shift
    input_row["demand"] = p
    input_row["social_buzz"] = input_row["social_buzz"] * (1 + np.random.normal(0,0.05))
    input_row["promo"] = 1 if np.random.rand() < 0.08 else 0
    input_row["weather_index"] = np.random.normal(0,1)

future_dates = pd.date_range(df_sku_prep["date"].max()+pd.Timedelta(days=1), periods=days_ahead)
forecast_df = pd.DataFrame({"date":future_dates, "forecast": np.round(preds,0)})
historical_plot = df_sku_prep[["date","demand"]].copy()
fig = px.line(historical_plot, x="date", y="demand", title=f"Historical demand for {sku_choice}")
fig.add_scatter(x=forecast_df["date"], y=forecast_df["forecast"], mode="lines+markers", name="Forecast")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Forecast Confidence (simple proxy)")
st.progress(min(100, max(10, int(100 - df_sku_prep["demand"].std()))))

st.markdown("## Inventory Digital Twin (simplified)")
# Simple fabricated warehouse inventory snapshot
warehouses = {
    "Dubai": {"WH":"WH_DXB","stock": {sku_choice: int(np.random.randint(0,500))}},
    "Riyadh": {"WH":"WH_RYD","stock": {sku_choice: int(np.random.randint(0,500))}},
    "Mumbai": {"WH":"WH_MUM","stock": {sku_choice: int(np.random.randint(0,500))}},
    "HoChiMinh": {"WH":"WH_SGN","stock": {sku_choice: int(np.random.randint(0,500))}},
}
inv_df = pd.DataFrame([{"warehouse":k, "sku": sku_choice, "qty": v["stock"][sku_choice]} for k,v in warehouses.items()])
st.table(inv_df)

# Reallocation suggestion
low_wh = inv_df.sort_values("qty").iloc[0]
high_wh = inv_df.sort_values("qty", ascending=False).iloc[0]
needed = int(max(0, forecast_df["forecast"].sum()/days_ahead*7 - low_wh["qty"]))
suggest = f"Recommend transfer {needed} units from {high_wh['warehouse']} ({high_wh['qty']} qty) to {low_wh['warehouse']} ({low_wh['qty']} qty) to cover 7 days expected demand."
st.info(suggest)

st.markdown("## Returns Management — AI grading demo")
# Simulate incoming returns and grading
returns = pd.DataFrame({
    "return_id": [f"R{1000+i}" for i in range(6)],
    "reason": np.random.choice(["Size","Damaged","Changed Mind","Wrong Item","Quality"], 6),
    "score": np.random.randint(30,100,6)
})
def grade_action(score, reason):
    if score>75 and reason!="Damaged": return "Resell"
    if reason=="Damaged": return "Recycle"
    if score>50: return "Refurbish"
    return "Liquidate"
returns["action"] = returns.apply(lambda r: grade_action(r["score"], r["reason"]), axis=1)
st.table(returns)

st.markdown("## Product Categorization & Segmentation")
# Use simple KMeans on SKU-level aggregated features
agg = df.groupby("sku").agg({"demand":["mean","std"], "social_buzz":"mean"})
agg.columns = ["d_mean","d_std","buzz"]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xk = sc.fit_transform(agg)
kmeans = KMeans(n_clusters=4, random_state=42).fit(Xk)
agg["cluster"] = kmeans.labels_
st.dataframe(agg.sort_values("d_mean", ascending=False).head(10))

st.markdown("## Trend Radar (social buzz top SKUs)")
buzz = df.groupby("sku")["social_buzz"].mean().reset_index().sort_values("social_buzz", ascending=False).head(6)
st.bar_chart(buzz.set_index("sku"))

st.markdown("### Export / Actions")
st.button("Export Forecast to CSV")
st.button("Create Reallocation Order (mock)")

st.markdown("---")
st.caption("Demo prototype built for presentation. Model and data are synthetic — replace with real ERP/WMS/social feeds for production.")
