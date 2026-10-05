import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STUDENT = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"
N = 8

print(f"Student: {STUDENT}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {N}")
print("City: Poltava")

np.random.seed(N)

city = "Полтава"
base_temp = 8.5
amplitude = 14
base_consumption = 1100
k = 28
noise_sd = 30

months = np.arange(1, 13)

temp = (
    base_temp
    + amplitude * np.cos((months - 7) / 12 * 2 * np.pi)
)

temp = np.round(
    temp + np.random.normal(0, 0.5, size=12),
    1
)

consumption = (
    base_consumption
    - k * (temp - base_temp)
    + np.random.normal(0, noise_sd, size=12)
)

consumption = np.round(consumption, 0)

city_data = pd.DataFrame({
    "місяць": months,
    "температура": temp,
    "споживання_МВтгод": consumption
})

print("\n--- Dataset ---")
print(city_data)

print("\n--- Task 1 ---")

print("Модель: споживання = a + b * температура")
print("Тип за випадковістю: стохастична")
print("Тип за призначенням: описова/прогнозна статистична")
print("Тип за залежністю від часу: статична")

print("\n--- Task 2 ---")

b, a = np.polyfit(
    city_data["температура"],
    city_data["споживання_МВтгод"],
    1
)

print(
    f"споживання ≈ {a:.3f} + ({b:.3f}) * температура"
)

city_data["передбачення"] = (
    a + b * city_data["температура"]
)

print("\nКоефіцієнт b:")
print(f"{b:.3f}")

print("\n--- Task 3 ---")

city_data["залишок"] = (
    city_data["споживання_МВтгод"]
    - city_data["передбачення"]
)

print(
    city_data[
        [
            "місяць",
            "температура",
            "споживання_МВтгод",
            "передбачення",
            "залишок"
        ]
    ]
)

ss_res = (
    city_data["залишок"] ** 2
).sum()

ss_tot = (
    (
        city_data["споживання_МВтгод"]
        - city_data["споживання_МВтгод"].mean()
    ) ** 2
).sum()

r_squared = 1 - ss_res / ss_tot

print(f"\nSS residual = {ss_res:.3f}")
print(f"SS total = {ss_tot:.3f}")
print(f"R² = {r_squared:.6f}")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    city_data["температура"],
    city_data["споживання_МВтгод"],
    label="Спостереження"
)

x_line = np.linspace(
    city_data["температура"].min(),
    city_data["температура"].max(),
    100
)

y_line = a + b * x_line

ax.plot(
    x_line,
    y_line,
    label="Лінійна модель"
)

ax.set_xlabel("Температура, °C")
ax.set_ylabel("Споживання, МВт·год")
ax.set_title("Лінійна регресія — Полтава")
ax.legend()

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    city_data["місяць"],
    city_data["залишок"]
)

ax.axhline(
    0,
    linestyle="--"
)

ax.set_xlabel("Місяць")
ax.set_ylabel("Залишок, МВт·год")
ax.set_title("Графік залишків — Полтава")

plt.tight_layout()
plt.show()

print("\n--- Task 4 ---")

first_temp = city_data.loc[0, "температура"]
first_prediction = city_data.loc[0, "передбачення"]

manual_prediction = a + b * first_temp

print(f"Температура першого місяця: {first_temp:.1f} °C")
print(f"Передбачення за кодом: {first_prediction:.3f}")
print(f"Ручний розрахунок: {manual_prediction:.3f}")
print(
    f"Різниця: "
    f"{abs(first_prediction - manual_prediction):.10f}"
)

print("\n--- Task 5 ---")

print(
    "За отриманими даними модель має дуже високе R² "
    "і залишки без вираженої систематичної структури."
)

print(
    "Для поставленої мети ускладнення моделі "
    "не є необхідним."
)

# Контрольні питання
#
# 1. Чим відрізняються верифікація та валідація?
#
# Верифікація перевіряє, чи правильно реалізована модель. Наприклад, чи
# правильно код обчислює a + b * температура.
#
# Валідація перевіряє, чи сама модель адекватно описує реальність. Для цього
# бажано порівнювати її прогнози з новими незалежними даними.
#
# 2. За якими ознаками класифікується модель?
#
# Модель можна незалежно класифікувати за кількома ознаками:
# - за наявністю випадковості — стохастична;
# - за характером задачі — описова/прогнозна статистична;
# - за залежністю від попередніх станів — статична.
# Ці ознаки не суперечать одна одній, тому що відповідають на різні питання
# про модель.
#
# 3. Що показує R² і чи доводить високе R², що модель правильна?
#
# R² показує частку варіації залежної змінної, яку пояснює модель. У нашому
# випадку R² ≈ 0.995, тобто приблизно 99.55% варіації споживання пояснюється
# температурою.
#
# Але високе R² саме по собі не доводить, що модель правильна або придатна для
# екстраполяції. Потрібно також перевіряти залишки та оцінювати модель на
# нових даних.
#
# 4. Чому структура залишків важливіша за одну сумарну величину помилок?
#
# R² та сума квадратів залишків показують загальний рівень помилки, але не
# показують, як саме ця помилка розподілена.
#
# Якщо залишки випадково розкидані навколо нуля — це аргумент на користь
# адекватності моделі. Якщо вони утворюють тренд, дугу або іншу структуру, це
# може свідчити про пропущену нелінійність або фактор, який модель не
# враховує. Саме такий підхід вимагається у вашому Завданні 3.