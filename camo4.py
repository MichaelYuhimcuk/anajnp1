"""СР 4. Дискретні розподіли в задачах (набір: якість повітря, Beijing).

Задача 1 (біноміальна): з n=10 випадково вибраних днів скільки буде днів
                        з небезпечним повітрям (добовий PM2.5 > 75 мкг/м3)?
Задача 2 (Пуассон):     скільки днів із сильним забрудненням (добовий PM2.5 > 150)
                        трапиться за один тиждень?

Вхід: dataset_tidy.csv.  Вихід: консоль + sr4_output/discrete.png, sr4_report.md
Запуск: pip install pandas numpy scipy matplotlib ; python sr4_discrete.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

STATION = "Dongsi"      # будь-яка станція з набору
THRESH_BINOM = 75       # мкг/м3, межа "небезпечного" дня
THRESH_POIS = 150       # мкг/м3, межа "сильного забруднення"
N, K = 10, 3            # біноміальна: 10 днів, рівно 3 небезпечні
os.makedirs("sr4_output", exist_ok=True)

# ---------- підготовка: добові середні PM2.5 однієї станції ----------
df = pd.read_csv("dataset_tidy.csv", parse_dates=["datetime"])
df = df[df["station"] == STATION].set_index("datetime")
g = df["pm25"].resample("D")
daily = g.mean()[g.count() >= 18].dropna()      # день рахуємо, якщо >= 18 годин даних
print(f"Станція {STATION}: {len(daily)} днів із достатніми даними")

# ================= Задача 1: біноміальний розподіл =================
p = (daily > THRESH_BINOM).mean()               # оцінка ймовірності "успіху"
b = stats.binom(N, p)
p_k = b.pmf(K)                                  # P(X = K)
p_ge = b.sf(K - 1)                              # P(X >= K)
mean_b = N * p                                  # E(X) = n p
var_b = N * p * (1 - p)                         # Var(X) = n p (1-p)

print("\n=== Задача 1: біноміальна ===")
print(f"p (частка днів PM2.5 > {THRESH_BINOM}) = {p:.3f}")
print(f"P(X = {K}) = {p_k:.4f}")
print(f"P(X >= {K}) = {p_ge:.4f}")
print(f"E(X) = n*p = {mean_b:.3f};  Var(X) = n*p*(1-p) = {var_b:.3f}")
print(f"Перевірка scipy: mean={b.mean():.3f}, var={b.var():.3f}")

# перевірка симуляцією: випадкові вибірки по N днів
rng = np.random.default_rng(1)
sims = np.array([(rng.choice(daily.values, N, replace=False) > THRESH_BINOM).sum()
                 for _ in range(20000)])
print(f"Симуляція: P(X={K}) ≈ {(sims == K).mean():.4f}, "
      f"E ≈ {sims.mean():.3f}, Var ≈ {sims.var():.3f}")

# ================= Задача 2: розподіл Пуассона =================
strong = (daily > THRESH_POIS).astype(int)
weekly = strong.resample("W").sum()             # кількість таких днів за тиждень
weekly = weekly[strong.resample("W").count() == 7]   # лише повні тижні
lam = weekly.mean()                             # оцінка lambda
pois = stats.poisson(lam)
k_show = 2
print("\n=== Задача 2: Пуассон ===")
print(f"Повних тижнів: {len(weekly)}, lambda = {lam:.3f} днів/тиждень")
print(f"P(X = 0) = {pois.pmf(0):.4f}")
print(f"P(X = {k_show}) = {pois.pmf(k_show):.4f}")
print(f"P(X >= 3) = {pois.sf(2):.4f}")
disp = weekly.var() / weekly.mean()
print(f"Індекс дисперсії Var/mean = {disp:.2f} (для Пуассона ≈ 1)")

# порівняння емпіричних частот з теоретичними
kmax = int(weekly.max())
emp = weekly.value_counts(normalize=True).reindex(range(kmax + 1), fill_value=0)
theo = pd.Series(pois.pmf(range(kmax + 1)), index=range(kmax + 1))
cmp_df = pd.DataFrame({"empirical": emp.round(4), "poisson": theo.round(4)})
print("\nЕмпіричні частоти vs Пуассон:\n", cmp_df)

# ================= Графіки =================
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ks = np.arange(N + 1)
axes[0].bar(ks, b.pmf(ks), color="steelblue")
axes[0].bar(K, b.pmf(K), color="tab:red", label=f"k={K}: {p_k:.3f}")
axes[0].set_xlabel("Кількість небезпечних днів із 10")
axes[0].set_ylabel("Імовірність")
axes[0].set_title(f"Біноміальний: n={N}, p={p:.2f}")
axes[0].set_xticks(ks)
axes[0].legend(frameon=False)

ks2 = np.arange(kmax + 1)
w = 0.4
axes[1].bar(ks2 - w / 2, emp.values, w, color="0.7", label="дані")
axes[1].bar(ks2 + w / 2, theo.values, w, color="steelblue",
            label=f"Пуассон, λ={lam:.2f}")
axes[1].set_xlabel("Днів із PM2.5 > 150 за тиждень")
axes[1].set_ylabel("Частка тижнів / імовірність")
axes[1].set_title("Розподіл Пуассона проти даних")
axes[1].set_xticks(ks2)
axes[1].legend(frameon=False)
fig.tight_layout()
fig.savefig("sr4_output/discrete.png", dpi=150)

# ================= Звіт (текст із підставленими числами) =================
report = f"""# СР 4. Дискретні розподіли

Набір: Beijing Multi-Site Air Quality, станція {STATION}, {len(daily)} днів
(добове середнє PM2.5, день враховано, якщо є >= 18 годинних вимірювань).

## Задача 1 (біноміальна)
З {N} випадково вибраних днів скільки матимуть добовий PM2.5 > {THRESH_BINOM} мкг/м3?
Яка ймовірність рівно {K} таких днів?

- p = {p:.3f} (частка таких днів у даних)
- P(X = {K}) = binom.pmf({K}, {N}, {p:.3f}) = **{p_k:.4f}**
- P(X >= {K}) = **{p_ge:.4f}**
- E(X) = n*p = {N}*{p:.3f} = **{mean_b:.2f}**
- Var(X) = n*p*(1-p) = **{var_b:.2f}**

**Чому біноміальна:** (1) кількість спроб фіксована (n = {N}); (2) кожна спроба має два
наслідки (день небезпечний / ні); (3) ймовірність успіху однакова (p); (4) спроби
незалежні - дні обираються випадково з усього періоду, а не поспіль (сусідні дні
пов'язані погодою, тому вибір послідовних днів порушив би незалежність).

## Задача 2 (Пуассон)
Скільки днів із добовим PM2.5 > {THRESH_POIS} мкг/м3 трапляється за тиждень?

- lambda = {lam:.3f} днів на тиждень (середнє за {len(weekly)} повних тижнів)
- P(X = 0) = **{pois.pmf(0):.4f}**, P(X = {k_show}) = **{pois.pmf(k_show):.4f}**,
  P(X >= 3) = **{pois.sf(2):.4f}**
- Індекс дисперсії Var/mean = {disp:.2f}

**Чому Пуассон:** рахуємо кількість подій за фіксований інтервал (тиждень);
події відносно рідкісні; середня інтенсивність стала. **Обмеження:** забруднення
тримається кілька днів поспіль (події не зовсім незалежні), тому індекс дисперсії
{'більший за 1 (надмірна дисперсія): реальні дані "кластеризуються", і Пуассон лише наближення' if disp > 1.2 else 'близький до 1: Пуассон добре описує дані'}.
"""
with open("sr4_output/sr4_report.md", "w", encoding="utf-8") as f:
    f.write(report)
print("\nЗбережено: sr4_output/discrete.png, sr4_output/sr4_report.md")
