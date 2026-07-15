"""Ad hoc: plot Close vs Trend LOWESS frac=0.02 saja (tanpa perbandingan kandidat frac lain).
Sumber data: crude-oil-forecasting-thesis/evaluations/lowess-eia1986-2026/lowess-fullseries-candidates.csv
"""
import pandas as pd
import matplotlib.pyplot as plt

SOURCE_CSV = r"D:\Coding\#bigdata\crude-oil-forecasting-thesis\evaluations\lowess-eia1986-2026\lowess-fullseries-candidates.csv"
OUTPUT_PATH = "evaluations/graphical/lowess/05_trend_only_frac_0.02.png"

df = pd.read_csv(SOURCE_CSV, parse_dates=["Date"])

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["Date"], df["Close"], label="Close (Actual)", color="lightgray", linewidth=1, alpha=0.9)
ax.plot(df["Date"], df["Trend_frac_0.02"], label="LOWESS Trend (frac=0.02)", color="crimson", linewidth=1.3)
ax.set_title("WTI Close vs LOWESS Trend (frac=0.02, Parameter Final)")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved: {OUTPUT_PATH}")
