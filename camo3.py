"""СР 3. Описова статистика і візуалізація власного набору (якість повітря).

Вхід:  dataset_tidy.csv (створений скриптом sr2_air_quality.py)
Вихід: папка sr3_output/ з графіками (PNG) і таблицями (CSV), підсумки в консолі.

Запуск:  pip install pandas matplotlib numpy
         python sr3_descriptive.py
Скрипт розбито на комірки "# %%" - його можна відкрити в VS Code / Jupyter.
"""
# %% 0. Налаштування та завантаження
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "sr3_output"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "axes.spines.top": False,      # без зайвих рамок (chartjunk)
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "figure.dpi": 100,
})

df = pd.read_csv("dataset_tidy.csv", parse_dates=["datetime"])
df["season"] = pd.Categorical(
    df["season"], categories=["winter", "spring", "summer", "autumn"],
    ordered=True)
aqi_order = ["Good", "Moderate", "Unhealthy for sensitive",
             "Unhealthy", "Very unhealthy", "Hazardous"]
df["aqi_category"] = pd.Categorical(df["aqi_category"],
                                    categories=aqi_order, ordered=True)
print("Розмір набору:", df.shape)

# %% 1. Числові змінні: describe + форма розподілу + викиди
# назва: (підпис осі з одиницями, колір)
NUM = {
    "pm25": "PM2.5, мкг/м³",
    "temperature": "Температура, °C",
    "wind_speed": "Швидкість вітру, м/с",
}

rows = []
for col, label in NUM.items():
    s = df[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((s < low) | (s > high)).sum())
    rows.append({
        "variable": col, "n": len(s), "mean": s.mean(), "median": s.median(),
        "std": s.std(), "min": s.min(), "q1": q1, "q3": q3, "max": s.max(),
        "iqr": iqr, "cv_%": s.std() / s.mean() * 100 if s.mean() else np.nan,
        "skew": s.skew(), "kurtosis": s.kurt(),
        "outliers_iqr": n_out, "outliers_%": n_out / len(s) * 100,
    })
stats = pd.DataFrame(rows).set_index("variable").round(2)
stats.to_csv(f"{OUT}/numeric_stats.csv")
print("\n=== Описові статистики ===\n", df[list(NUM)].describe().round(2))
print("\n=== Форма розподілу і викиди ===\n", stats[
    ["mean", "median", "skew", "kurtosis", "outliers_iqr", "outliers_%"]])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (col, label) in zip(axes, NUM.items()):
    s = df[col].dropna()
    ax.hist(s, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(s.mean(), color="red", ls="--", lw=1.5,
               label=f"середнє {s.mean():.1f}")
    ax.axvline(s.median(), color="black", ls=":", lw=1.5,
               label=f"медіана {s.median():.1f}")
    ax.set_xlabel(label)
    ax.set_ylabel("Кількість спостережень")
    ax.set_title(f"Розподіл: {col}")
    ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/hist_numeric.png", dpi=150)

# boxplot-и для наочності викидів
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, (col, label) in zip(axes, NUM.items()):
    ax.boxplot(df[col].dropna(), vert=False, showfliers=True,
               flierprops=dict(markersize=2, alpha=0.3))
    ax.set_xlabel(label)
    ax.set_yticks([])
    ax.set_title(col)
    ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(f"{OUT}/box_numeric.png", dpi=150)

# %% 2. Категоріальні змінні: частотні таблиці + стовпчикові діаграми
CAT = {"season": "Пора року", "aqi_category": "Категорія якості повітря"}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, (col, label) in zip(axes, CAT.items()):
    counts = df[col].value_counts(sort=False)          # порядок категорій
    freq = pd.DataFrame({"count": counts,
                         "share_%": (counts / counts.sum() * 100).round(1)})
    freq.to_csv(f"{OUT}/freq_{col}.csv")
    print(f"\n=== Частоти: {col} ===\n", freq)

    bars = ax.bar(freq.index.astype(str), freq["count"], color="steelblue")
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in freq["share_%"]],
                 fontsize=8, padding=2)
    ax.set_ylim(0, freq["count"].max() * 1.12)           # вісь від нуля
    ax.set_xlabel(label)
    ax.set_ylabel("Кількість спостережень")
    ax.set_title(f"Частоти: {col}")
    ax.tick_params(axis="x", rotation=25 if col == "aqi_category" else 0)
fig.tight_layout()
fig.savefig(f"{OUT}/bar_categorical.png", dpi=150)

# %% 3. groupby + agg: PM2.5 за станціями
by_station = (
    df.groupby("station", observed=True)["pm25"]
      .agg(n="count", mean="mean", median="median", std="std",
           p90=lambda s: s.quantile(0.9))
      .round(1)
      .sort_values("mean", ascending=False)
)
by_station.to_csv(f"{OUT}/pm25_by_station.csv")
print("\n=== PM2.5 за станціями ===\n", by_station)

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(by_station.index, by_station["mean"], color="steelblue",
              yerr=by_station["std"] / np.sqrt(by_station["n"]) * 1.96,
              capsize=3)
ax.bar_label(bars, fmt="%.0f", padding=6, fontsize=8)
ax.set_ylim(0, by_station["mean"].max() * 1.15)          # від нуля
ax.set_xlabel("Станція")
ax.set_ylabel("Середній PM2.5, мкг/м³")
ax.set_title("Середній PM2.5 за станціями (вуса: 95% ДІ середнього)")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(f"{OUT}/pm25_by_station.png", dpi=150)

# додатковий розріз: за порою року кілька показників
by_season = (
    df.groupby("season", observed=True)
      .agg(pm25_mean=("pm25", "mean"), pm25_median=("pm25", "median"),
           temp_mean=("temperature", "mean"),
           wind_mean=("wind_speed", "mean"), n=("pm25", "size"))
      .round(1)
)
by_season.to_csv(f"{OUT}/by_season.csv")
print("\n=== Показники за порою року ===\n", by_season)

# %% 4. Зведені таблиці: pivot_table і crosstab
# 4a. середній PM2.5: станція x пора року
pivot = df.pivot_table(values="pm25", index="station", columns="season",
                       aggfunc="mean", observed=True).round(1)
pivot.to_csv(f"{OUT}/pivot_pm25_station_season.csv")
print("\n=== pivot_table: середній PM2.5 (станція x пора року) ===\n", pivot)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(pivot.shape[1]), pivot.columns)
ax.set_yticks(range(pivot.shape[0]), pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center",
                fontsize=8)
ax.grid(False)
cb = fig.colorbar(im, ax=ax)
cb.set_label("Середній PM2.5, мкг/м³")
ax.set_xlabel("Пора року")
ax.set_ylabel("Станція")
ax.set_title("Середній PM2.5: станція × пора року")
fig.tight_layout()
fig.savefig(f"{OUT}/pivot_heatmap.png", dpi=150)

# 4b. crosstab: частка категорій якості повітря в кожну пору року, %
ct = (pd.crosstab(df["season"], df["aqi_category"], normalize="index") * 100
      ).round(1)
ct.to_csv(f"{OUT}/crosstab_season_aqi.csv")
print("\n=== crosstab: частка категорій якості повітря за порою року, % ===\n",
      ct)

fig, ax = plt.subplots(figsize=(9, 4.5))
bottom = np.zeros(len(ct))
colors = ["#2ca02c", "#bcbd22", "#ff7f0e", "#d62728", "#9467bd", "#7f2d2d"]
for cat, c in zip(ct.columns, colors):
    ax.bar(ct.index.astype(str), ct[cat], bottom=bottom, color=c, label=cat)
    bottom += ct[cat].values
ax.set_ylim(0, 100)
ax.set_xlabel("Пора року")
ax.set_ylabel("Частка спостережень, %")
ax.set_title("Структура якості повітря за порами року")
ax.legend(title="Категорія", bbox_to_anchor=(1.02, 1), loc="upper left",
          frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/crosstab_stacked.png", dpi=150)

# %% 5. Автоматичні висновки (числа беруться з даних)
def shape_word(skew):
    if abs(skew) < 0.5:
        return "приблизно симетричний"
    return "скошений вправо (довгий правий хвіст)" if skew > 0 \
        else "скошений вліво (довгий лівий хвіст)"

print("\n=== ВИСНОВКИ ===")
for col in NUM:
    r = stats.loc[col]
    print(f"- {col}: середнє {r['mean']}, медіана {r['median']}; розподіл "
          f"{shape_word(r['skew'])} (skew={r['skew']}); викидів за IQR: "
          f"{int(r['outliers_iqr'])} ({r['outliers_%']:.1f}%).")

worst, best = by_station["mean"].idxmax(), by_station["mean"].idxmin()
print(f"- Найзабрудненіша станція за середнім PM2.5: {worst} "
      f"({by_station.loc[worst, 'mean']}), найчистіша: {best} "
      f"({by_station.loc[best, 'mean']}).")
s_worst = by_season["pm25_mean"].idxmax()
s_best = by_season["pm25_mean"].idxmin()
print(f"- Найвищий середній PM2.5 в сезон '{s_worst}' "
      f"({by_season.loc[s_worst, 'pm25_mean']}), найнижчий в '{s_best}' "
      f"({by_season.loc[s_best, 'pm25_mean']}).")
print(f"\nГрафіки й таблиці збережено в папці {OUT}/")
