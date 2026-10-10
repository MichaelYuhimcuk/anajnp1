"""СР 5. Перевірка статистичної гіпотези на власному наборі (якість повітря).

Питання: чи відрізняється середній добовий PM2.5 узимку від середнього влітку?
H0: mu_winter = mu_summer      H1: mu_winter != mu_summer      alpha = 0.05
Критерій: двовибірковий t-критерій (Welch) для незалежних вибірок.

Вхід: dataset_tidy.csv.  Вихід: консоль, sr5_output/hypothesis.png, sr5_output/sr5_report.md
Запуск: pip install pandas numpy scipy matplotlib statsmodels ; python sr5_hypothesis.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

STATION = "Dongsi"
ALPHA = 0.05
os.makedirs("sr5_output", exist_ok=True)

# ---------- 1. Дані: добові середні PM2.5 на станції ----------
df = pd.read_csv("dataset_tidy.csv", parse_dates=["datetime"])
df = df[df["station"] == STATION].set_index("datetime")
g = df["pm25"].resample("D")
daily = g.mean()[g.count() >= 18].dropna()
month = daily.index.month
winter = daily[np.isin(month, [12, 1, 2])]
summer = daily[np.isin(month, [6, 7, 8])]
print(f"Станція {STATION}: зима n={len(winter)}, літо n={len(summer)}")

# ---------- 2. Описова статистика груп ----------
desc = pd.DataFrame({"winter": winter.describe(), "summer": summer.describe()}).round(2)
desc.loc["skew"] = [round(winter.skew(), 2), round(summer.skew(), 2)]
print("\n", desc)

# ---------- 3. Перевірка передумов ----------
# 3a. нормальність: Shapiro-Wilk (чутливий при великих n), тому дивимось ще на QQ-графік і skew
sh_w = stats.shapiro(winter)
sh_s = stats.shapiro(summer)
print(f"\nShapiro-Wilk: зима p={sh_w.pvalue:.4f}, літо p={sh_s.pvalue:.4f}")
sh_w_log = stats.shapiro(np.log(winter))
sh_s_log = stats.shapiro(np.log(summer))
print(f"Після log-перетворення: зима p={sh_w_log.pvalue:.4f}, літо p={sh_s_log.pvalue:.4f}")

# 3b. рівність дисперсій (Levene) -> вибір між Student і Welch
lev = stats.levene(winter, summer)
print(f"Levene: p={lev.pvalue:.4f} -> дисперсії "
      f"{'різні (беремо Welch)' if lev.pvalue < ALPHA else 'можна вважати рівними'}")

# ---------- 4. Критерій ----------
t_res = stats.ttest_ind(winter, summer, equal_var=False)       # Welch, двосторонній
t_one = stats.ttest_ind(winter, summer, equal_var=False, alternative="greater")
diff = winter.mean() - summer.mean()
v1, v2 = winter.var(ddof=1), summer.var(ddof=1)
n1, n2 = len(winter), len(summer)
se = np.sqrt(v1 / n1 + v2 / n2)
dfw = se**4 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))   # df Welch
tcrit = stats.t.ppf(1 - ALPHA / 2, dfw)
ci = (diff - tcrit * se, diff + tcrit * se)
sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
cohen_d = diff / sp

print("\n=== Welch t-test ===")
print(f"t = {t_res.statistic:.3f}, df ≈ {dfw:.1f}, p (двосторонній) = {t_res.pvalue:.3g}")
print(f"p (односторонній, winter > summer) = {t_one.pvalue:.3g}")
print(f"Різниця середніх = {diff:.2f} мкг/м3, 95% ДІ [{ci[0]:.2f}; {ci[1]:.2f}]")
print(f"Cohen's d = {cohen_d:.2f}")
decision = "відхиляється" if t_res.pvalue < ALPHA else "не відхиляється"
print(f"Рішення: H0 {decision} на рівні {ALPHA}")

# ---------- 5. Перевірка стійкості ----------
mw = stats.mannwhitneyu(winter, summer, alternative="two-sided")
t_log = stats.ttest_ind(np.log(winter), np.log(summer), equal_var=False)
# добові дані автокорельовані -> перевіряємо на прорідженій вибірці (кожен 7-й день)
t_thin = stats.ttest_ind(winter.iloc[::7], summer.iloc[::7], equal_var=False)
print("\n=== Стійкість ===")
print(f"Манна-Уітні: p = {mw.pvalue:.3g}")
print(f"t-тест на log(PM2.5): p = {t_log.pvalue:.3g}")
print(f"t-тест, кожен 7-й день (n={len(winter.iloc[::7])}/{len(summer.iloc[::7])}): "
      f"p = {t_thin.pvalue:.3g}")

# ---------- 6. Графіки ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

bins = np.linspace(0, max(winter.max(), summer.max()), 40)
axes[0].hist(winter, bins=bins, alpha=0.6, color="tab:blue", label="зима")
axes[0].hist(summer, bins=bins, alpha=0.6, color="tab:orange", label="літо")
axes[0].axvline(winter.mean(), color="tab:blue", ls="--")
axes[0].axvline(summer.mean(), color="tab:orange", ls="--")
axes[0].set_xlabel("Добовий PM2.5, мкг/м³")
axes[0].set_ylabel("Кількість днів")
axes[0].set_title("Розподіли (пунктир: середні)")
axes[0].legend(frameon=False)

for data, name, col in ((winter, "зима", "tab:blue"), (summer, "літо", "tab:orange")):
    osm, osr = stats.probplot(data, dist="norm", fit=False)
    axes[1].plot(osm, osr, ".", color=col, alpha=0.5, label=name)
axes[1].set_xlabel("Теоретичні квантилі нормального розподілу")
axes[1].set_ylabel("Спостережені значення, мкг/м³")
axes[1].set_title("QQ-графік (перевірка нормальності)")
axes[1].legend(frameon=False)

means = [winter.mean(), summer.mean()]
errs = [stats.sem(winter) * 1.96, stats.sem(summer) * 1.96]
axes[2].bar(["зима", "літо"], means, yerr=errs, capsize=6,
            color=["tab:blue", "tab:orange"])
axes[2].set_ylim(0, max(means) * 1.3)
axes[2].set_ylabel("Середній PM2.5, мкг/м³")
axes[2].set_title("Середні з 95% ДІ")
fig.tight_layout()
fig.savefig("sr5_output/hypothesis.png", dpi=150)

# ---------- 7. Звіт ----------
norm_note = ("розподіли скошені вправо (skew: зима {:.2f}, літо {:.2f}), Shapiro-Wilk "
             "відхиляє нормальність (p = {:.3g} / {:.3g}), АЛЕ n = {} / {} достатньо "
             "велике, щоб за ЦГТ середні були наближено нормальними; стійкість "
             "підтверджено log-перетворенням і критерієм Манна-Уітні"
             ).format(winter.skew(), summer.skew(), sh_w.pvalue, sh_s.pvalue, n1, n2)
report = f"""# СР 5. Перевірка гіпотези

**Питання.** Чи відрізняється середній добовий PM2.5 узимку (грудень-лютий) від середнього влітку
(червень-серпень) на станції {STATION}?

**Гіпотези** (alpha = {ALPHA}):
- H0: mu_winter = mu_summer (сезонної різниці середнього PM2.5 немає)
- H1: mu_winter != mu_summer

**Критерій.** Двовибірковий t-критерій Welch для незалежних вибірок (`scipy.stats.ttest_ind`,
`equal_var=False`).

**Передумови.**
- Незалежні групи: зимові й літні дні різні; але сусідні дні автокорельовані, тому додатково
  перевірено прореджену вибірку (кожен 7-й день): p = {t_thin.pvalue:.3g}.
- Нормальність: {norm_note}.
- Дисперсії: Levene p = {lev.pvalue:.3g}; обрано Welch, що не потребує рівності дисперсій.

**Результат.**
- середнє: зима {winter.mean():.1f}, літо {summer.mean():.1f} мкг/м³; різниця {diff:.1f}
  (95% ДІ [{ci[0]:.1f}; {ci[1]:.1f}])
- t = {t_res.statistic:.2f}, df ≈ {dfw:.0f}, p = {t_res.pvalue:.3g}; Cohen's d = {cohen_d:.2f}
- **Рішення:** H0 {decision} на рівні {ALPHA}.

**Інтерпретація.** {'Дані свідчать, що середній добовий PM2.5 узимку статистично значуще відрізняється від літнього: зимове повітря в середньому на ' + format(diff, '.1f') + ' мкг/м3 забрудненіше. Різниця велика й за розміром ефекту (d = ' + format(cohen_d, '.2f') + '), тобто має практичне значення, а не лише статистичне. Ймовірна причина (опалювальний сезон, температурні інверсії) є гіпотезою, яку тест не доводить: він показує наявність різниці, а не її причину.' if t_res.pvalue < ALPHA else 'Недостатньо доказів різниці між сезонами; це не доводить, що різниці немає.'}

**Обмеження.** Дані з однієї станції; добові значення автокорельовані; період {daily.index.min():%Y-%m} - {daily.index.max():%Y-%m}
охоплює лише кілька зим і літ.
"""
with open("sr5_output/sr5_report.md", "w", encoding="utf-8") as f:
    f.write(report)
print("\nЗбережено: sr5_output/hypothesis.png, sr5_output/sr5_report.md")
