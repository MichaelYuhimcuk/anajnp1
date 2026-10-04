import numpy as np
from scipy import stats

STUDENT_NAME = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"
VARIANT = 8
CITY = "Poltava"

JULY_TEMP = 21
MU0 = 19.6
N = 30
ALPHA = 0.05

print(f"Student: {STUDENT_NAME}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")
print(f"City: {CITY}")
print(f"July temperature for generation: {JULY_TEMP} C")
print(f"Old norm mu0: {MU0} C")

np.random.seed(VARIANT)

daily_temps = np.random.normal(
    loc=JULY_TEMP,
    scale=2.5,
    size=N
)

x_bar = daily_temps.mean()
s = daily_temps.std(ddof=1)

print("\n--- Task 1 ---")
print("Daily temperatures:")
print(daily_temps)

print(f"\nSample mean x_bar = {x_bar:.4f} C")
print(f"Sample standard deviation s = {s:.4f} C")

print("\n--- Task 2 ---")

print("H0: mu = 19.6 C")
print("H1: mu != 19.6 C")
print("Test type: two-sided")

print("\n--- Task 3 ---")

se = s / np.sqrt(N)

z = (x_bar - MU0) / se

p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print(f"Standard error = {se:.4f}")
print(f"z-statistic = {z:.4f}")
print(f"p-value = {p_value:.6f}")

print("\n--- Task 4 ---")

print(f"Alpha = {ALPHA}")

if p_value < ALPHA:
    print("Decision: reject H0")
    print(
        f"The data provide statistically significant evidence "
        f"that the mean July temperature in {CITY} differs "
        f"from the old norm of {MU0} C."
    )
else:
    print("Decision: do not reject H0")
    print(
        f"The data do not provide sufficient evidence that "
        f"the mean July temperature in {CITY} differs "
        f"from the old norm of {MU0} C."
    )

print("\n--- Task 5 ---")

ci_lower = x_bar - 1.96 * se
ci_upper = x_bar + 1.96 * se

print(f"95% confidence interval: ({ci_lower:.4f}, {ci_upper:.4f}) C")

if ci_lower <= MU0 <= ci_upper:
    print("The old norm mu0 is inside the confidence interval.")
else:
    print("The old norm mu0 is outside the confidence interval.")

print("\nType I error:")
print(
    "Rejecting H0 even though the old norm of 19.6 C is actually correct."
)

print("\nType II error:")
print(
    "Not rejecting H0 even though the true mean July temperature "
    "is actually different from 19.6 C."
)

# **1. Що означає отримане p-value?**
#
# У нашому дослідженні p-value ≈ 0.0053 означає, що якби стара норма μ = 19.6°C була правильною, то отримати вибіркове середнє з таким або ще більшим відхиленням від 19.6°C було б досить малоймовірно — приблизно у 0.53% випадків за припущенням H0.
#
# p-value не є ймовірністю того, що стара норма правильна або неправильна.
#
# **2. Чому тип тесту потрібно обирати до перегляду результату?**
#
# Тип тесту повинен визначатися самим дослідницьким питанням, а не отриманими даними. У нашій роботі нас цікавить будь-яке відхилення від старої норми, тому заздалегідь обрано двобічну альтернативу H1: μ ≠ 19.6°C.
#
# Якби тип тесту обирали після перегляду вибіркового середнього, це могло б штучно збільшити ймовірність отримання статистично значущого результату.
#
# **3. Що означають помилки I та II роду?**
#
# Помилка I роду — це відхилення H0, коли стара норма 19.6°C насправді правильна. У такому випадку ми помилково заявили б, що кліматична норма змінилася.
#
# Помилка II роду — це ситуація, коли H0 не відхиляється, хоча справжня середня температура вже відрізняється від старої норми. Тоді ми помилково залишили б застарілу норму.
#
# **4. Чи узгоджується тест із довірчим інтервалом?**
#
# Так. Тест дав p-value ≈ 0.0053, що менше за α = 0.05, тому H0 відхиляється.
#
# 95% довірчий інтервал дорівнює приблизно (20.06°C; 22.25°C) і не містить μ0 = 19.6°C. Тому обидва методи дають однаковий висновок.
#
# Це очікувано, оскільки для двобічного тесту при α = 0.05 перевірка H0 еквівалентна перевірці того, чи належить μ0 95% довірчому інтервалу.