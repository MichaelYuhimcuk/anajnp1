import numpy as np
from scipy import stats
import pandas as pd

STUDENT_NAME = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"
VARIANT = 8
CITY = "Poltava"
JULY_TEMP = 21

print(f"Student: {STUDENT_NAME}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")
print(f"City: {CITY}")
print(f"July temperature: {JULY_TEMP} C")

print("\n--- Task 1 ---")

np.random.seed(VARIANT)

daily_temps = np.random.normal(
    loc=JULY_TEMP,
    scale=2.5,
    size=30
)

mean = daily_temps.mean()
s = daily_temps.std(ddof=1)
n = len(daily_temps)

print("Daily temperatures:")
print(daily_temps)

print(f"\nMean: {mean:.4f} C")
print(f"Sample standard deviation: {s:.4f} C")
print(f"Sample size: {n}")

print("\n--- Task 2 ---")

se = s / np.sqrt(n)

ci_mean_95 = stats.t.interval(
    0.95,
    df=n - 1,
    loc=mean,
    scale=se
)

print(f"Standard error: {se:.4f}")
print(f"95% confidence interval: ({ci_mean_95[0]:.4f}, {ci_mean_95[1]:.4f}) C")

print("\n--- Task 3 ---")

above_july = daily_temps > JULY_TEMP
p_hat = above_july.mean()

n_p = n * p_hat
n_1p = n * (1 - p_hat)

print(f"Proportion above {JULY_TEMP} C: {p_hat:.4f}")
print(f"n * p_hat: {n_p:.4f}")
print(f"n * (1 - p_hat): {n_1p:.4f}")

if n_p >= 5 and n_1p >= 5:
    print("Normal approximation condition is satisfied.")

    se_p = np.sqrt(p_hat * (1 - p_hat) / n)

    ci_prop = stats.norm.interval(
        0.95,
        loc=p_hat,
        scale=se_p
    )

    print(f"Standard error of proportion: {se_p:.4f}")
    print(
        f"95% confidence interval for proportion: "
        f"({ci_prop[0]:.4f}, {ci_prop[1]:.4f})"
    )

    if ci_prop[0] <= 0.5 <= ci_prop[1]:
        print("The theoretical value 0.5 is inside the interval.")
    else:
        print("The theoretical value 0.5 is outside the interval.")
else:
    print("Normal approximation condition is NOT satisfied.")

print("\n--- Task 4 ---")

confidence_levels = [0.90, 0.95, 0.99]
intervals = []
widths = []

for confidence in confidence_levels:
    interval = stats.t.interval(
        confidence,
        df=n - 1,
        loc=mean,
        scale=se
    )

    width = interval[1] - interval[0]

    intervals.append(interval)
    widths.append(width)

results = pd.DataFrame({
    "Confidence level": ["90%", "95%", "99%"],
    "Lower bound": [x[0] for x in intervals],
    "Upper bound": [x[1] for x in intervals],
    "Width": widths
})

print(results.to_string(index=False))

print("\n--- Task 5 ---")

print(
    f"95% confidence interval for the population mean: "
    f"({ci_mean_95[0]:.4f}, {ci_mean_95[1]:.4f}) C"
)

print(
    "If we repeated the sampling procedure many times and built "
    "a 95% confidence interval each time, approximately 95% of "
    "those intervals would contain the true population mean."
)

# ## Контрольні питання
#
# **1. Яка різниця між точковою оцінкою і довірчим інтервалом?**
#
# Точкова оцінка — це одне число, яке використовується як оцінка невідомого параметра. Наприклад, вибіркове середнє `x̄` є точковою оцінкою генерального середнього.
#
# Довірчий інтервал дає не одне число, а діапазон можливих значень параметра з урахуванням випадкової похибки вибірки. Тому одного лише `x̄` недостатньо для повної оцінки, оскільки він не показує, наскільки точно отримано оцінку.
#
# **2. Чому використовують t-розподіл, а не нормальний?**
#
# Коли σ генеральної сукупності невідоме, його замінюють вибірковим стандартним відхиленням `s`. Через додаткову невизначеність використовується t-розподіл із `n - 1` ступенями вільності. Для невеликих вибірок t-розподіл має ширші хвости, ніж стандартний нормальний розподіл.
#
# **3. Що насправді означає 95% довіри?**
#
# 95% довіри означає властивість процедури побудови інтервалу. Якщо багато разів отримувати незалежні вибірки однакового розміру та кожного разу будувати 95% довірчий інтервал однаковим методом, приблизно 95% таких інтервалів міститимуть істинне значення параметра.
#
# Не слід говорити, що для конкретного вже побудованого інтервалу існує 95% ймовірність містити істинне значення.
#
# **4. Які три фактори визначають ширину довірчого інтервалу?**
#
# Основними факторами є:
#
# 1. **Рівень довіри** — чим він вищий, тим ширший інтервал.
# 2. **Розкид даних** — більша стандартна похибка або стандартне відхилення збільшує ширину інтервалу.
# 3. **Розмір вибірки** — збільшення `n` зменшує стандартну похибку та звужує інтервал.
#
# Отже, для отримання точнішої оцінки можна збільшити розмір вибірки або зменшити варіативність вимірювань.