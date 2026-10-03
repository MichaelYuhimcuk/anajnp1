import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STUDENT_NAME = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"

VARIANT = 8
CITY = "Poltava"
BASE_SALES = 34
BASE_TEMP = 1

np.random.seed(VARIANT)

days = np.arange(1, 31)

weekend = np.isin(days % 7, [6, 0])

temperature = np.round(
    BASE_TEMP + np.random.uniform(-3, 3, size=30),
    1
)

sales_variation = np.random.randint(-6, 7, size=30)

sales = BASE_SALES + sales_variation

weekend_percent = np.random.uniform(0.15, 0.25, size=30)

sales = np.where(
    weekend,
    sales * (1 + weekend_percent),
    sales
)

sales = np.round(sales).astype(int)

data = pd.DataFrame({
    "day": days,
    "weekend": weekend,
    "temperature": temperature,
    "sales": sales
})

print(f"Student: {STUDENT_NAME}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")
print(f"City: {CITY}")

print("\n--- Task 1 ---")
print(data)

fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(
    data["sales"],
    bins=5,
    edgecolor="black"
)

ax.set_xlabel("Sales, units per day")
ax.set_ylabel("Number of days")
ax.set_title(
    f"Distribution of daily sales in {CITY}"
)

ax.set_ylim(bottom=0)

fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(
    data["sales"],
    bins=8,
    edgecolor="black"
)

ax.set_xlabel("Sales, units per day")
ax.set_ylabel("Number of days")
ax.set_title(
    f"Distribution of daily sales in {CITY}, bins=8"
)

ax.set_ylim(bottom=0)

weekday_sales = data.loc[
    ~data["weekend"],
    "sales"
]

weekend_sales = data.loc[
    data["weekend"],
    "sales"
]

fig, ax = plt.subplots(figsize=(6, 4))

ax.boxplot(
    [weekday_sales, weekend_sales],
    labels=["Weekdays", "Weekends"]
)

ax.set_ylabel("Sales, units per day")
ax.set_title(
    f"Sales comparison: weekdays and weekends in {CITY}"
)

fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(
    data["temperature"],
    data["sales"]
)

ax.set_xlabel("Temperature, °C")
ax.set_ylabel("Sales, units per day")
ax.set_title(
    f"Daily sales versus temperature in {CITY}"
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(
    data["sales"],
    bins=5,
    edgecolor="black"
)

axes[0].set_xlabel("Sales, units per day")
axes[0].set_ylabel("Number of days")
axes[0].set_title("Correct histogram")
axes[0].set_ylim(bottom=0)

axes[1].hist(
    data["sales"],
    bins=5,
    edgecolor="black"
)

axes[1].set_xlabel("Sales, units per day")
axes[1].set_ylabel("Number of days")
axes[1].set_title("Distorted histogram")

max_count = np.histogram(data["sales"], bins=5)[0].max()
axes[1].set_ylim(max_count * 0.45, max_count * 1.05)

plt.tight_layout()

plt.show()

# Контрольні питання
# 
# 1. Чому обрізана вісь Y перебільшує різницю?
# Тому що графік перестає починатися з нуля. Навіть невелика числова різниця між стовпцями займає більшу частину доступної висоти графіка, тому візуально різниця здається значно більшою.
# 
# 2. Коли лінійний графік є помилкою для категоріальних даних?
# Наприклад, якщо порівнюються продажі різних типів напоїв: Tea, Coffee, Juice. З'єднання цих категорій лінією створює враження, що між ними існує послідовний числовий порядок. Для номінальної шкали такого порядку немає, тому краще використати стовпчикову діаграму.
#
# 3. Чому підписи осей, одиниці та заголовок є частиною графіка?
# Вони пояснюють, що саме показують дані, у яких одиницях вони вимірюються та який зміст має графік. Без цих елементів читач може неправильно зрозуміти представлену інформацію.
#
# 4. Що таке chartjunk?
# Chartjunk — це зайві декоративні елементи графіка, які не допомагають зрозуміти дані. Наприклад, непотрібний 3D-ефект, велика кількість декоративних кольорів або об'ємні стовпці. Вони ускладнюють порівняння значень і відволікають увагу від даних.