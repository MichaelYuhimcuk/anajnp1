import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr

N = 8
city = "Poltava"

monthly_temp = [-5, -4, 1, 10, 16, 19, 21, 20, 14, 8, 1, -3]

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

np.random.seed(N)

rows = []

for month, temp in zip(months, monthly_temp):
    heating_days = np.clip(
        18 - temp + np.random.normal(0, 1.5),
        0,
        30
    )
    heating_days = round(heating_days)

    cost = 0.15 * heating_days + np.random.normal(0, 1.0)
    cost = round(max(cost, 0), 1)

    rows.append({
        "city": city,
        "month": month,
        "temperature": temp,
        "heating_days": heating_days,
        "heating_cost": cost
    })

climate = pd.DataFrame(rows)

print("Student: Mykola Yukhymchuk")
print("Group: IT-41")
print("Variant: 8")
print("City: Poltava")
print()
print(climate)

print("\n--- Task 1 ---")

x = climate["temperature"].to_numpy()
y = climate["heating_cost"].to_numpy()

result = linregress(x, y)

b1 = result.slope
b0 = result.intercept

print(f"Slope b1 = {b1:.3f}")
print(f"Intercept b0 = {b0:.3f}")
print(f"R^2 = {result.rvalue ** 2:.3f}")
print(f"p-value = {result.pvalue:.6f}")

print(
    f"Regression equation: heating_cost = "
    f"{b0:.3f} + ({b1:.3f}) * temperature"
)

print("\n--- Task 2 ---")

predicted = b0 + b1 * x
residuals = y - predicted

ss_residual = np.sum(residuals ** 2)
ss_total = np.sum((y - np.mean(y)) ** 2)

r2_method_1 = result.rvalue ** 2
r2_method_2 = 1 - ss_residual / ss_total

r, pearson_p = pearsonr(x, y)

print(f"R^2 by rvalue^2 = {r2_method_1:.6f}")
print(f"SS residual = {ss_residual:.6f}")
print(f"SS total = {ss_total:.6f}")
print(f"R^2 by SS formula = {r2_method_2:.6f}")
print(f"Pearson r = {r:.6f}")
print(f"r^2 = {r ** 2:.6f}")
print(f"Pearson p-value = {pearson_p:.6f}")

print("\n--- Task 3 ---")

climate["predicted_cost"] = predicted
climate["residual"] = residuals

print(
    climate[
        [
            "month",
            "temperature",
            "heating_cost",
            "predicted_cost",
            "residual"
        ]
    ]
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.scatter(
    climate["temperature"],
    climate["residual"]
)

ax.axhline(
    0,
    linestyle="--"
)

ax.set_xlabel("Temperature, °C")
ax.set_ylabel("Residual, thousand UAH")
ax.set_title("Residual plot — Poltava")

plt.tight_layout()
plt.show()

print("\n--- Task 4 ---")

temperature_new = -10
prediction = b0 + b1 * temperature_new

min_temp = x.min()
max_temp = x.max()

print(f"Observed temperature range: {min_temp} ... {max_temp} °C")
print(f"Prediction temperature: {temperature_new} °C")
print(f"Predicted heating cost = {prediction:.3f} thousand UAH")

if min_temp <= temperature_new <= max_temp:
    print("The prediction is inside the observed range.")
else:
    print("The prediction is an extrapolation outside the observed range.")

print("\n--- Task 5 ---")

print(f"linregress p-value = {result.pvalue:.10f}")
print(f"pearsonr p-value   = {pearson_p:.10f}")
print(f"Difference         = {abs(result.pvalue - pearson_p):.10f}")

# Контрольні питання
#
# 1. Чому метод найменших квадратів мінімізує саме суму квадратів залишків?
#
# Квадрати роблять усі відхилення додатними, тому позитивні та негативні
# залишки не можуть взаємно компенсуватися. Крім того, великі помилки
# отримують більшу вагу, а функція суми квадратів є зручною для математичної
# оптимізації.
#
# 2. Що означає коефіцієнт нахилу b1 і чим він відрізняється від кореляції r?
#
# b1 показує, наскільки в середньому змінюються витрати на опалення при зміні
# температури на 1°C. Він має конкретні одиниці вимірювання — тис. грн/°C.
# Коефіцієнт кореляції r показує силу та напрямок лінійного зв'язку і є
# безрозмірним числом від -1 до 1.
#
# 3. Чому високий R² не гарантує адекватності моделі?
#
# R² показує лише частку поясненої варіації. Він не показує, чи є
# нелінійність, систематичні закономірності або неоднакова величина залишків.
# Це можна побачити на графіку залишків.
#
# 4. Що відбувається з надійністю прогнозу за межами діапазону?
#
# Прогноз за межами спостережених значень є екстраполяцією. Він менш
# надійний, оскільки модель не перевірялася на таких значеннях x, і реальна
# залежність за межами вибірки може відрізнятися від побудованої прямої.