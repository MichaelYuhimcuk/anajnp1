import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

STUDENT_NAME = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"

VARIANT = 8
CITY = "Poltava"
JULY_TEMP = 21

np.random.seed(VARIANT)

daily_temps = np.random.normal(
    loc=JULY_TEMP,
    scale=2.5,
    size=30
)

print(f"Student: {STUDENT_NAME}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")
print(f"City: {CITY}")

print("\n--- Task 1 ---")
print("Daily temperatures:")
print(np.round(daily_temps, 2))

print(f"\nSample mean: {daily_temps.mean():.2f}")
print(f"Sample std: {daily_temps.std():.2f}")

print("\n--- Task 2 ---")

mean = daily_temps.mean()
std = daily_temps.std()

dist = stats.norm(
    loc=mean,
    scale=std
)

print(f"Mean (mu): {mean:.2f}")
print(f"Standard deviation (sigma): {std:.2f}")

print(
    f"Normal distribution: "
    f"N({mean:.2f}, {std:.2f})"
)

print("\n--- Task 3 ---")

x = np.linspace(
    daily_temps.min() - 3,
    daily_temps.max() + 3,
    200
)

fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(
    daily_temps,
    bins=8,
    density=True,
    edgecolor="black",
    alpha=0.7
)

ax.plot(
    x,
    dist.pdf(x),
    linewidth=2
)

ax.set_xlabel("Temperature, °C")
ax.set_ylabel("Density")
ax.set_title(
    f"Daily temperature distribution in {CITY}"
)

plt.show()

print("\n--- Task 4 ---")

lower = mean - std
upper = mean + std

probability = (
    dist.cdf(upper) -
    dist.cdf(lower)
)

print(f"Lower bound: {lower:.2f} °C")
print(f"Upper bound: {upper:.2f} °C")
print(
    f"Probability: {probability:.4f}"
)
print(
    f"Percentage: {probability * 100:.2f}%"
)

print("\n--- Task 5 ---")

percentile_95 = dist.ppf(0.95)

print(
    f"95th percentile: "
    f"{percentile_95:.2f} °C"
)

print(
    f"This means that approximately 95% "
    f"of temperatures in the fitted normal "
    f"model are below {percentile_95:.2f} °C."
)

# ## Контрольні питання
#
# **1. Різниця між `pdf()` і `cdf()`**
#
# `pdf(x)` повертає **значення густини** розподілу в точці `x`. Вона відповідає на питання: *наскільки висока теоретична густина розподілу в цій точці?*
#
# `cdf(x)` повертає **накопичену ймовірність**:
#
# > яка ймовірність отримати значення, не більше за `x`?
#
# Наприклад:
#
# ```
# dist.cdf(20)
# ```
#
# повертає ймовірність того, що температура буде не більшою за 20 °C.
#
# ---
#
# **2. Чому `pdf(x)` може бути більшим за 1?**
#
# `pdf()` повертає не ймовірність, а **густину ймовірності**. Густина може бути більшою за 1.
#
# Ймовірність визначається **площею під кривою** на певному інтервалі, а не висотою кривої в одній точці. Загальна площа під PDF для всього розподілу дорівнює 1.
#
# ---
#
# **3. Чому `ppf()` є оберненою до `cdf()`?**
#
# `cdf()` працює за принципом:
#
# ```
# значення → накопичена ймовірність
# ```
#
# Наприклад:
#
# ```
# dist.cdf(20)
# ```
#
# питає: *яка частка значень знаходиться нижче 20 °C?*
#
# `ppf()` працює навпаки:
#
# ```
# накопичена ймовірність → значення
# ```
#
# Наприклад:
#
# ```
# dist.ppf(0.95)
# ```
#
# питає: *яка температура є межею, нижче якої знаходиться 95% значень?*