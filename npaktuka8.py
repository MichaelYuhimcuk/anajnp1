import numpy as np
import pandas as pd

STUDENT_NAME = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"

VARIANT = 8
CITY = "Poltava"
BASE_TEMP = 8.5
AMPLITUDE = 14

np.random.seed(42)

rows = []

for year in [2021, 2022, 2023, 2024]:
    for month in range(1, 13):

        seasonal = AMPLITUDE * np.cos(
            (month - 7) / 12 * 2 * np.pi
        )

        noise = np.random.normal(0, 1.0)

        rows.append({
            "city": CITY,
            "year": year,
            "month": month,
            "temperature": round(
                BASE_TEMP + seasonal + noise,
                1
            )
        })

climate = pd.DataFrame(rows)

print(f"Student: {STUDENT_NAME}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")
print(f"City: {CITY}")

print("\n--- Task 1 ---")
print(climate)

print("\n--- Task 1.1: groupby + agg by year ---")

year_stats = climate.groupby("year")["temperature"].agg(
    ["mean", "min", "max"]
)

print(year_stats)

print("\n--- Task 2 ---")

month_stats = climate.groupby("month")["temperature"].agg(
    ["mean", "std"]
)

print(month_stats)

most_unstable_month = month_stats["std"].idxmax()
largest_std = month_stats["std"].max()

print(
    f"\nMonth with the largest temperature variation: "
    f"{most_unstable_month}"
)

print(
    f"Standard deviation: {largest_std:.2f}"
)

print("\n--- Task 3 ---")

temperature_pivot = climate.pivot_table(
    index="month",
    columns="year",
    values="temperature",
    aggfunc="mean"
)

print(temperature_pivot)

print("\n--- Task 4 ---")

print("\n--- Task 4 ---")

climate["season"] = np.select(
    [
        climate["month"].isin([12, 1, 2]),
        climate["month"].isin([3, 4, 5]),
        climate["month"].isin([6, 7, 8]),
        climate["month"].isin([9, 10, 11])
    ],
    [
        "Winter",
        "Spring",
        "Summer",
        "Autumn"
    ],
    default="Unknown"
)

climate["warmer_than_average"] = (
    climate["temperature"] > BASE_TEMP
)

print(climate)

print("\nCrosstab:")

season_crosstab = pd.crosstab(
    climate["season"],
    climate["warmer_than_average"]
)

print(season_crosstab)

print("\n--- Task 5 ---")

pivot_result = climate.pivot(
    index="month",
    columns="year",
    values="temperature"
)

print(pivot_result)

print(
    "\nPivot successfully completed because "
    "each month/year combination occurs only once."
)

print("\n--- Dataset information ---")

print(f"Number of rows: {len(climate)}")
print(f"Number of columns: {len(climate.columns)}")

print("\nFirst rows:")
print(climate.head())

# Контрольні питання
# 
# 1. Логіка split-apply-combine
# 
# groupby() працює за принципом:
# 
# Split — дані розділяються на групи за певною ознакою.
# Apply — до кожної групи застосовується операція, наприклад mean(), min(), max() або std().
# Combine — результати об'єднуються в одну таблицю або Series.
#
# Наприклад:
# 
# climate.groupby("year")["temperature"].mean()
#
# спочатку розділяє дані за роками, потім знаходить середнє для кожного року і повертає об'єднаний результат.
#
# 2. Відмінність pivot() і pivot_table()
#
# pivot() вимагає, щоб кожна комбінація індексу та стовпця відповідала одному значенню. Якщо є дублікати, виникає помилка.
#
# pivot_table() може працювати з дублікованими комбінаціями, оскільки вона додатково виконує агрегацію, наприклад:
#
# aggfunc="mean"
#
# або:
#
# aggfunc="sum"
#
# 3. Що показує crosstab()?
#
# crosstab() створює таблицю частот для категоріальних значень. Вона показує, скільки разів зустрічається кожна комбінація категорій.
#
# Наприклад:
#
# pd.crosstab(
#     climate["season"],
#     climate["warmer_than_average"]
# )
#
# покаже кількість спостережень для кожного сезону та значення True/False.
#
# Еквівалентний підрахунок можна зробити через:
#
# climate.groupby(
#     ["season", "warmer_than_average"]
# ).size()
#
# Але crosstab() одразу представляє результат у зручній перехресній таблиці.
#
# 4. Чому для groupby() і pivot_table() зручний tidy-формат?
#
# У tidy-форматі кожна змінна є окремим стовпцем, а кожне спостереження — окремим рядком. Це дозволяє легко групувати дані, фільтрувати їх, виконувати агрегації та будувати зведені таблиці.
# 
# Наприклад:
# 
# city | year | month | temperature
#
# є зручнішою структурою для аналізу, ніж таблиця, де кожен рік або місяць записаний у назві окремого стовпця.