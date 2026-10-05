import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import probplot, shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor

STUDENT = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"
N = 8

print(f"Student: {STUDENT}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {N}")
print("City: Poltava")

np.random.seed(42)

city = "Полтава"
mean_temp = 8.5
amplitude = 14
hum_base = 72

rows = []

for month in range(1, 13):

    temp = (
        mean_temp
        + amplitude * np.cos((month - 7) / 12 * 2 * np.pi)
        + np.random.normal(0, 1.0)
    )

    humidity = hum_base + np.random.normal(0, 5.0)

    if temp < 8:
        opal_days = 30
    elif temp < 12:
        opal_days = 15
    else:
        opal_days = 0

    cost = (
        200
        - 9.0 * temp
        + 0.6 * humidity
        + 2.5 * opal_days
        + np.random.normal(0, 15)
    )

    rows.append({
        "місто": city,
        "місяць": month,
        "температура": round(temp, 1),
        "вологість": round(humidity, 1),
        "опалювальні_дні": opal_days,
        "витрати_на_опалення": round(cost, 1)
    })

heating = pd.DataFrame(rows)

print("\n--- Dataset ---")
print(heating)

print("\n--- Task 1 ---")

X = heating[["температура", "вологість"]]
X = sm.add_constant(X)

y = heating["витрати_на_опалення"]

model = sm.OLS(y, X).fit()

print(model.summary())

print("\nR-squared:")
print(f"{model.rsquared:.6f}")

print("\nAdjusted R-squared:")
print(f"{model.rsquared_adj:.6f}")

print("\n--- Task 2 ---")

print("Coefficients:")
print(model.params)

print("\nP-values:")
print(model.pvalues)

print("\nConfidence intervals:")
print(model.conf_int())

print("\n--- Task 3 ---")

vif_table = pd.DataFrame()

vif_table["variable"] = X.columns
vif_table["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print(vif_table)

print("\nVIF for predictors:")

for i, column in enumerate(X.columns):

    if column != "const":
        vif = variance_inflation_factor(X.values, i)
        print(f"{column}: {vif:.3f}")

print("\n--- Task 4 ---")

X3 = heating[
    [
        "температура",
        "вологість",
        "опалювальні_дні"
    ]
]

X3 = sm.add_constant(X3)

model3 = sm.OLS(y, X3).fit()

print(model3.summary())

print("\nComparison of models:")

comparison = pd.DataFrame({
    "Модель": [
        "2 предиктори",
        "3 предиктори"
    ],
    "R2": [
        model.rsquared,
        model3.rsquared
    ],
    "Adjusted R2": [
        model.rsquared_adj,
        model3.rsquared_adj
    ]
})

print(comparison)

print("\nVIF for 3-predictor model:")

for i, column in enumerate(X3.columns):

    if column != "const":
        vif = variance_inflation_factor(X3.values, i)
        print(f"{column}: {vif:.3f}")

print("\n--- Task 5 ---")

fitted = model.fittedvalues
residuals = model.resid

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(fitted, residuals)

ax.axhline(
    0,
    linestyle="--"
)

ax.set_xlabel("Передбачені витрати на опалення")
ax.set_ylabel("Залишки")
ax.set_title("Графік залишків — модель з двома предикторами")

plt.tight_layout()
plt.show()


# QQ-plot
fig, ax = plt.subplots(figsize=(7, 5))

probplot(
    residuals,
    dist="norm",
    plot=ax
)

ax.set_title("QQ-plot залишків")

plt.tight_layout()
plt.show()


# Тест Шапіро-Вілка
shapiro_stat, shapiro_p = shapiro(residuals)

print(f"Shapiro-Wilk statistic = {shapiro_stat:.6f}")
print(f"Shapiro-Wilk p-value = {shapiro_p:.6f}")

if shapiro_p < 0.05:
    print("Відхиляємо H0: залишки не мають нормального розподілу.")
else:
    print("Не відхиляємо H0: немає достатніх підстав вважати залишки ненормальними.")

print("\n--- Task 6 ---")

X_simple = sm.add_constant(
    heating[["температура"]]
)

simple_model = sm.OLS(
    y,
    X_simple
).fit()

print("Simple regression:")
print(f"R2 = {simple_model.rsquared:.6f}")
print(f"Adjusted R2 = {simple_model.rsquared_adj:.6f}")

print("\nMultiple regression:")
print(f"R2 = {model.rsquared:.6f}")
print(f"Adjusted R2 = {model.rsquared_adj:.6f}")

print("\nChange in R2:")
print(
    f"{model.rsquared - simple_model.rsquared:.6f}"
)

print("\nChange in adjusted R2:")
print(
    f"{model.rsquared_adj - simple_model.rsquared_adj:.6f}"
)

# Контрольні питання
#
# 1. Чому sm.add_constant() потрібний перед sm.OLS()?
#
# sm.add_constant() додає до матриці предикторів стовпець одиниць. Він
# відповідає вільному члену b0 у рівнянні:
# y = b0 + b1*x1 + b2*x2.
# Якщо не додати constant, модель за замовчуванням не матиме окремого
# вільного члена і буде змушена проходити через початок координат. Це може
# суттєво змінити коефіцієнти та якість моделі.
#
# 2. Чим коефіцієнт множинної регресії відрізняється від коефіцієнта простої
# регресії?
#
# У простій регресії коефіцієнт показує зв'язок між результатом і одним
# предиктором без контролю інших змінних.
#
# У множинній регресії коефіцієнт показує зміну залежної змінної при зміні
# конкретного предиктора на одну одиницю за інших рівних умов, тобто при
# незмінних інших предикторах.
#
# 3. Чому звичайний R² не може зменшитися при додаванні нового предиктора?
#
# Метод найменших квадратів при додаванні нового предиктора отримує більше
# можливостей підібрати модель до даних. Тому сума квадратів залишків не може
# збільшитися, а R² не може зменшитися. Навіть абсолютно непотрібний предиктор
# може трохи підвищити R².
#
# Тому для порівняння моделей із різною кількістю предикторів корисніше
# використовувати Adjusted R².
#
# 4. Що означає VIF ≈ 1 та VIF > 10?
#
# VIF ≈ 1 означає, що предиктор практично не має лінійної залежності з іншими
# предикторами.
#
# VIF > 10 свідчить про сильну мультиколінеарність. У такому випадку
# коефіцієнти можуть бути нестабільними, стандартні помилки — завеликими, а
# інтерпретація окремих предикторів — ненадійною. На практиці варто перевірити
# пов'язані предиктори та розглянути вилучення одного з них або інший спосіб
# побудови моделі.