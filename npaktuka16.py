import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

STUDENT = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"
VARIANT = 8
CITY = "Poltava"

print(f"Student: {STUDENT}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")
print(f"City: {CITY}")

np.random.seed(VARIANT)

monthly_temp = [
    -5, -4, 1, 10, 16, 19,
    21, 20, 14, 8, 1, -3
]

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

rows = []

for month, temp in zip(months, monthly_temp):

    heating_days = np.clip(
        18 - temp + np.random.normal(0, 1.5),
        0,
        30
    )

    heating_days = round(heating_days)

    cost = (
        0.15 * heating_days
        + np.random.normal(0, 1.0)
    )

    cost = round(max(cost, 0), 1)

    rows.append({
        "city": CITY,
        "month": month,
        "temperature": temp,
        "heating_days": heating_days,
        "heating_cost": cost
    })


climate = pd.DataFrame(rows)


print("\n--- Generated climate data ---")
print(climate.to_string(index=False))

print("\n--- Task 1 ---")

fig, ax = plt.subplots()

ax.scatter(
    climate["temperature"],
    climate["heating_cost"]
)

ax.set_xlabel("Temperature, °C")
ax.set_ylabel("Heating cost, thousand UAH")
ax.set_title(
    f"Temperature and heating cost - {CITY}"
)

print(
    "The scatter plot shows a negative relationship between "
    "temperature and heating cost."
)

print(
    "When the temperature decreases, heating costs generally increase. "
    "This agrees with the logic of the generated data."
)

print(
    "The relationship is approximately linear, although random noise "
    "causes some deviations from a perfect straight line."
)

print("\n--- Task 2 ---")

temperature = climate["temperature"]
heating_cost = climate["heating_cost"]

pearson_result = pearsonr(
    temperature,
    heating_cost
)

spearman_result = spearmanr(
    temperature,
    heating_cost
)

print(
    f"Pearson: r = {pearson_result.statistic:.4f}, "
    f"p-value = {pearson_result.pvalue:.6f}"
)

print(
    f"Spearman: rho = {spearman_result.statistic:.4f}, "
    f"p-value = {spearman_result.pvalue:.6f}"
)


alpha = 0.05


if pearson_result.pvalue < alpha:
    print(
        "Pearson conclusion: reject H0. "
        "There is a statistically significant correlation."
    )
else:
    print(
        "Pearson conclusion: do not reject H0. "
        "There is not enough evidence of correlation."
    )


if spearman_result.pvalue < alpha:
    print(
        "Spearman conclusion: reject H0. "
        "There is a statistically significant monotonic relationship."
    )
else:
    print(
        "Spearman conclusion: do not reject H0. "
        "There is not enough evidence of a monotonic relationship."
    )


print(
    "\nPearson and Spearman are expected to be relatively close "
    "because the relationship between temperature and heating cost "
    "is generally monotonic and approximately linear."
)

confidence_interval = pearson_result.confidence_interval(
    confidence_level=0.95
)

print(
    f"\n95% confidence interval for Pearson r: "
    f"[{confidence_interval.low:.4f}, "
    f"{confidence_interval.high:.4f}]"
)

print(
    "The confidence interval gives a range of plausible values "
    "for the true population correlation."
)

print("\n--- Task 3 ---")

correlation_data = climate[
    [
        "temperature",
        "heating_days",
        "heating_cost"
    ]
]

pearson_matrix = correlation_data.corr()

print("\nPearson correlation matrix:")
print(pearson_matrix.round(4))


print("\nSpearman correlation matrix:")

spearman_matrix = correlation_data.corr(
    method="spearman"
)

print(spearman_matrix.round(4))

pairs = [
    (
        "temperature",
        "heating_days",
        abs(pearson_matrix.loc["temperature", "heating_days"])
    ),
    (
        "temperature",
        "heating_cost",
        abs(pearson_matrix.loc["temperature", "heating_cost"])
    ),
    (
        "heating_days",
        "heating_cost",
        abs(pearson_matrix.loc["heating_days", "heating_cost"])
    )
]

strongest_pair = max(
    pairs,
    key=lambda item: item[2]
)

weakest_pair = min(
    pairs,
    key=lambda item: item[2]
)

print(
    f"\nStrongest correlation: "
    f"{strongest_pair[0]} - {strongest_pair[1]}, "
    f"|r| = {strongest_pair[2]:.4f}"
)

print(
    f"Weakest correlation: "
    f"{weakest_pair[0]} - {weakest_pair[1]}, "
    f"|r| = {weakest_pair[2]:.4f}"
)

print(
    "\nThe strong relationship between temperature and heating days "
    "is expected because heating days are calculated directly "
    "using temperature."
)

print(
    "Heating cost is also related to temperature because the cost "
    "is calculated from the number of heating days."
)

print("\n--- Task 4 ---")

print(
    "The strong correlation in this dataset is not simply an accidental "
    "association. The data generation formula directly uses temperature "
    "to calculate heating days, and heating days are then used to "
    "calculate heating cost."
)

print(
    "Therefore, there is a direct mathematical relationship in the "
    "generated dataset."
)

print(
    "This differs from the example of ice cream sales and drowning, "
    "where a third variable such as temperature can influence both "
    "variables and create a misleading correlation."
)

print(
    "In this practical work the relationship is intentionally built "
    "into the data generation formula."
)

print("\n--- Task 5 ---")

climate_outlier = climate.copy()

outlier = pd.DataFrame({
    "city": [CITY],
    "month": ["Anomaly"],
    "temperature": [15],
    "heating_days": [5],
    "heating_cost": [15.0]
})

climate_outlier = pd.concat(
    [climate_outlier, outlier],
    ignore_index=True
)


temperature_outlier = climate_outlier["temperature"]
cost_outlier = climate_outlier["heating_cost"]


pearson_outlier = pearsonr(
    temperature_outlier,
    cost_outlier
)

spearman_outlier = spearmanr(
    temperature_outlier,
    cost_outlier
)


print("\nOriginal Pearson correlation:")
print(
    f"r = {pearson_result.statistic:.4f}"
)

print("Pearson correlation with outlier:")
print(
    f"r = {pearson_outlier.statistic:.4f}"
)


print("\nOriginal Spearman correlation:")
print(
    f"rho = {spearman_result.statistic:.4f}"
)

print("Spearman correlation with outlier:")
print(
    f"rho = {spearman_outlier.statistic:.4f}"
)


pearson_change = abs(
    pearson_outlier.statistic -
    pearson_result.statistic
)

spearman_change = abs(
    spearman_outlier.statistic -
    spearman_result.statistic
)


print(
    f"\nChange in Pearson coefficient: "
    f"{pearson_change:.4f}"
)

print(
    f"Change in Spearman coefficient: "
    f"{spearman_change:.4f}"
)


if pearson_change > spearman_change:
    print(
        "Pearson changed more strongly than Spearman. "
        "This is expected because Pearson correlation is more "
        "sensitive to extreme values."
    )
else:
    print(
        "In this generated sample Spearman changed more strongly. "
        "The exact result depends on the position of the added outlier."
    )

plt.show()

# Контрольні запитання

# 1. Чому коефіцієнт Пірсона нормований діленням на добуток стандартних відхилень і завжди лежить у [-1, 1]?
# Коефіцієнт Пірсона показує силу та напрямок лінійного зв'язку між двома змінними.
# Ділення на добуток стандартних відхилень нормує результат, тому його значення
# завжди знаходиться в межах від -1 до 1.

# 2. Чому Спірмен працює з рангами і зазвичай стійкіший до викидів?
# Коефіцієнт Спірмена замість самих значень використовує їхні ранги.
# Тому окремі дуже великі або дуже малі значення мають менший вплив на результат,
# що робить цей метод більш стійким до викидів.

# 3. Наведіть приклад, коли r ≈ 0, але залежність сильна, проте нелінійна.
# Прикладом є залежність y = x^2. Якщо значення x симетрично розташовані навколо нуля,
# лінійна кореляція Пірсона може бути близькою до нуля, хоча залежність між змінними
# є сильною, але нелінійною.

# 4. Чому сильна кореляція не доводить причинність?
# Сильна кореляція показує лише наявність статистичного зв'язку між змінними.
# Вона не доводить, що одна змінна безпосередньо спричиняє зміну іншої,
# оскільки на результат можуть впливати інші фактори.