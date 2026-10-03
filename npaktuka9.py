import numpy as np
import pandas as pd

STUDENT_NAME = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"

VARIANT = 8
CITY = "Poltava"
BASE_PRICE = 700
BASE_QUANTITY = 4

customers = pd.DataFrame({
    "age": [19, 23, 28, 34, 41, 47, 53, 59, 64, 69],
    "income": [
        18000, 22000, 27000, 32000, 38000,
        45000, 52000, 61000, 70000, 82000
    ],
    "quantity": [2, 5, 3, 6, 4, 7, 5, 3, 8, 4],
    "delivery_city": [
        "Poltava",
        "Kyiv",
        "Lviv",
        "Odesa",
        "Kharkiv",
        "Poltava",
        "Dnipro",
        "Lviv",
        "Kyiv",
        "Odesa"
    ]
})

print(f"Student: {STUDENT_NAME}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")
print(f"City: {CITY}")

print("\n--- Task 1 ---")
print(customers)

print("\n--- Task 2 ---")

customers["order_total"] = (
    BASE_PRICE * customers["quantity"]
)

customers["income_share"] = (
    customers["order_total"] / customers["income"]
)

print(
    customers[
        [
            "age",
            "income",
            "quantity",
            "order_total",
            "income_share"
        ]
    ]
)

print("\n--- Task 3 ---")

customers["age_group_cut"] = pd.cut(
    customers["age"],
    bins=3
)

cut_counts = customers["age_group_cut"].value_counts().sort_index()

print("\nAge groups using cut():")
print(cut_counts)

customers["age_group_qcut"] = pd.qcut(
    customers["age"],
    q=3
)

qcut_counts = (
    customers["age_group_qcut"]
    .value_counts()
    .sort_index()
)

print("\nAge groups using qcut():")
print(qcut_counts)

print("\n--- Task 4 ---")

customers["age_minmax"] = (
    (customers["age"] - customers["age"].min())
    / (customers["age"].max() - customers["age"].min())
)

customers["income_minmax"] = (
    (customers["income"] - customers["income"].min())
    / (customers["income"].max() - customers["income"].min())
)

customers["age_z"] = (
    (customers["age"] - customers["age"].mean())
    / customers["age"].std()
)

customers["income_z"] = (
    (customers["income"] - customers["income"].mean())
    / customers["income"].std()
)

print("\nMin-max scaling:")
print(
    customers[
        [
            "age",
            "income",
            "age_minmax",
            "income_minmax"
        ]
    ].head()
)

print("\nZ-score standardization:")
print(
    customers[
        [
            "age",
            "income",
            "age_z",
            "income_z"
        ]
    ].head()
)

print("\n--- Task 5 ---")

city_dummies = pd.get_dummies(
    customers["delivery_city"],
    prefix="city"
)

customers = pd.concat(
    [customers, city_dummies],
    axis=1
)

print("\nOne-hot encoded cities:")
print(city_dummies)

print(
    f"\nUnique cities: "
    f"{customers['delivery_city'].nunique()}"
)

print(
    f"New city columns: "
    f"{len(city_dummies.columns)}"
)

print("\nFinal DataFrame:")
print(customers)

# ## Контрольні питання
#
# **1. Різниця між `cut()` і `qcut()`**
#
# `cut()` створює інтервали однакової довжини:
#
# ```
# pd.cut(data, bins=3)
# ```
#
# `qcut()` створює групи приблизно однакового розміру за кількістю спостережень:
#
# ```
# pd.qcut(data, q=3)
# ```
#
# Тому `cut()` підходить, коли важливі фіксовані діапазони значень, а `qcut()` — коли потрібно отримати приблизно однакову кількість об'єктів у кожній групі.
#
# **2. Чому для міста використовується one-hot?**
#
# Місто — номінальна категорія без природного порядку. Кодування `Kyiv = 1`, `Lviv = 2`, `Odesa = 3` створило б штучний порядок. One-hot кодування цього недоліку не має.
#
# **3. Навіщо масштабувати змінні?**
#
# Масштабування потрібне для методів, чутливих до величини змінних. Наприклад, у дистанційних методах змінна з великими числовими значеннями може непропорційно впливати на результат. Масштабування приводить змінні до порівнянного масштабу.
#
# **4. Навіщо потрібні похідні змінні?**
#
# Похідна змінна може краще відображати реальний зміст даних. Наприклад, `income_share` показує не просто суму покупки, а її відносний розмір порівняно з доходом клієнта. Це дозволяє коректніше порівнювати клієнтів із різними доходами.