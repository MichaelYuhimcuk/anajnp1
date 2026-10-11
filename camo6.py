"""СР 6. Кореляція та регресія на власному наборі (якість повітря, станція Dongsi).

1) Пара змінних: швидкість вітру -> PM2.5 (Пірсон, Спірмен, діаграма розсіювання)
2) Проста лінійна регресія log(PM2.5) ~ wind_speed
3) Множинна регресія log(PM2.5) ~ wind_speed + temperature + pressure + rain
4) Діагностика залишків, VIF, текстова інтерпретація

Вхід: dataset_tidy.csv.  Вихід: sr6_output/ (графіки, sr6_report.md)
Запуск: pip install pandas numpy scipy matplotlib statsmodels ; python sr6_regression.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

STATION = "Dongsi"
os.makedirs("sr6_output", exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

# ---------- дані: добові значення ----------
df = pd.read_csv("dataset_tidy.csv", parse_dates=["datetime"])
df = df[df["station"] == STATION].set_index("datetime")
daily = df.resample("D").agg({"pm25": "mean", "wind_speed": "mean",
                              "temperature": "mean", "pressure": "mean",
                              "rain": "sum"}).dropna()
daily["log_pm25"] = np.log(daily["pm25"])
print("Днів:", len(daily))
print(daily.corr(method="pearson").round(2))

# ---------- 1. Кореляція ----------
x, y = daily["wind_speed"], daily["pm25"]
r, p_r = stats.pearsonr(x, y)
rho, p_rho = stats.spearmanr(x, y)
r_log, _ = stats.pearsonr(x, daily["log_pm25"])
print(f"\nПірсон r={r:.3f} (p={p_r:.2g}); Спірмен rho={rho:.3f} (p={p_rho:.2g}); "
      f"Пірсон з log(PM2.5) r={r_log:.3f}")

# ---------- 2. Проста регресія ----------
X1 = sm.add_constant(daily[["wind_speed"]])
m1 = sm.OLS(daily["log_pm25"], X1).fit()
print("\n=== Проста регресія ===\n", m1.summary().tables[1], f"\nR2={m1.rsquared:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(x, y, s=8, alpha=0.4)
axes[0].set_xlabel("Швидкість вітру, м/с")
axes[0].set_ylabel("PM2.5, мкг/м³")
axes[0].set_title(f"Розсіювання (Пірсон {r:.2f}, Спірмен {rho:.2f})")
axes[1].scatter(x, daily["log_pm25"], s=8, alpha=0.4)
xs = np.linspace(x.min(), x.max(), 100)
axes[1].plot(xs, m1.params["const"] + m1.params["wind_speed"] * xs, color="red")
axes[1].set_xlabel("Швидкість вітру, м/с")
axes[1].set_ylabel("ln(PM2.5)")
axes[1].set_title(f"Проста регресія, R² = {m1.rsquared:.2f}")
fig.tight_layout()
fig.savefig("sr6_output/scatter_simple.png", dpi=150)

# ---------- 3. Множинна регресія ----------
PRED = ["wind_speed", "temperature", "pressure", "rain"]
Xm = sm.add_constant(daily[PRED])
m2 = sm.OLS(daily["log_pm25"], Xm).fit()
print("\n=== Множинна регресія ===\n", m2.summary())

vif = pd.Series([variance_inflation_factor(Xm.values, i) for i in range(1, Xm.shape[1])],
                index=PRED).round(2)
print("\nVIF (мультиколінеарність, >5-10 погано):\n", vif)

# ---------- 4. Залишки ----------
resid, fitted = m2.resid, m2.fittedvalues
dw = durbin_watson(resid)
bp = het_breuschpagan(resid, Xm)
sh = stats.shapiro(resid)
print(f"\nDurbin-Watson={dw:.2f} (≈2 немає автокореляції); "
      f"Breusch-Pagan p={bp[1]:.3g}; Shapiro p={sh.pvalue:.3g}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(fitted, resid, s=8, alpha=0.4)
axes[0].axhline(0, color="red")
axes[0].set_xlabel("Передбачене ln(PM2.5)")
axes[0].set_ylabel("Залишки")
axes[0].set_title("Залишки проти передбачень")
sm.qqplot(resid, line="s", ax=axes[1], markersize=3)
axes[1].set_title("QQ-графік залишків")
axes[2].plot(resid.index, resid, lw=0.6)
axes[2].set_xlabel("Дата")
axes[2].set_ylabel("Залишки")
axes[2].set_title("Залишки в часі")
fig.tight_layout()
fig.savefig("sr6_output/residuals.png", dpi=150)

# ---------- Звіт ----------
def pct(b):
    return (np.exp(b) - 1) * 100

rows = "\n".join(
    f"- `{k}`: b = {m2.params[k]:.4f}, p = {m2.pvalues[k]:.3g} -> зміна PM2.5 приблизно "
    f"{pct(m2.params[k]):+.1f}% на одиницю предиктора (за незмінних інших)"
    for k in PRED)
report = f"""# СР 6. Кореляція і регресія (станція {STATION}, добові дані, n = {len(daily)})

## Кореляція вітер - PM2.5
Пірсон r = {r:.2f}, Спірмен rho = {rho:.2f}. Зв'язок від'ємний: сильніший вітер розсіює
забруднення. Зв'язок нелінійний (PM2.5 скошений вправо, викиди), тому надійніше дивитися на
Спірмена та використовувати ln(PM2.5): r з log = {r_log:.2f}.

## Проста регресія ln(PM2.5) ~ вітер
ln(PM2.5) = {m1.params['const']:.2f} + ({m1.params['wind_speed']:.2f}) * вітер; R2 = {m1.rsquared:.2f}.
Кожен додатковий м/с вітру пов'язаний зі зміною PM2.5 приблизно на
{pct(m1.params['wind_speed']):+.0f}%. Вітер пояснює {m1.rsquared*100:.0f}% варіації ln(PM2.5).

## Множинна регресія
Залежна: ln(PM2.5). Предиктори: швидкість вітру (розсіювання), температура (опалювальний
сезон, інверсії), тиск (стійкі антициклони), опади (вимивання). Не взято PM10, SO2, NO2, CO
та ін.: вони самі наслідки забруднення (витік інформації про відповідь), а не незалежні
фактори; DEWP сильно колінеарна з температурою (мультиколінеарність).
R2 = {m2.rsquared:.2f}, скоригований R2 = {m2.rsquared_adj:.2f}.

{rows}

VIF: {vif.to_dict()}.

## Залишки
Durbin-Watson = {dw:.2f}: {'сильна автокореляція (добові дані залежні в часі), стандартні похибки і p-значення занадто оптимістичні' if dw < 1.5 else 'автокореляція слабка'}.
Breusch-Pagan p = {bp[1]:.3g}: {'гетероскедастичність' if bp[1] < 0.05 else 'гетероскедастичності не виявлено'}.
Нормальність залишків (Shapiro) p = {sh.pvalue:.3g}; докладніше див. residuals.png.

## Обережність інтерпретації
Модель показує асоціації, а не причинність; добові дані автокорельовані.
"""
with open("sr6_output/sr6_report.md", "w", encoding="utf-8") as f:
    f.write(report)
print("\nЗбережено в sr6_output/")
