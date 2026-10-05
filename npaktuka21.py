import numpy as np
import matplotlib.pyplot as plt

STUDENT = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"
VARIANT = 8

A = 0
B = 4
EXACT = 16 / 3


print(f"Student: {STUDENT}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")
print("City: Poltava")
print("Task: Integral from 0 to 4 of sqrt(x) dx")
print(f"Exact value: {EXACT:.6f}")

def monte_carlo_integral(n):
    np.random.seed(VARIANT)

    x = np.random.uniform(A, B, size=n)
    y = np.sqrt(x)

    estimate = (B - A) * y.mean()

    return estimate

print("\n--- Task 1 ---")

N = 1000

estimate_1000 = monte_carlo_integral(N)

print(f"N = {N}")
print(f"Monte Carlo estimate = {estimate_1000:.6f}")

print("\n--- Task 2 ---")

absolute_error_1000 = abs(estimate_1000 - EXACT)
relative_error_1000 = absolute_error_1000 / EXACT * 100

print(f"Monte Carlo estimate = {estimate_1000:.6f}")
print(f"Exact value = {EXACT:.6f}")
print(f"Absolute error = {absolute_error_1000:.6f}")
print(f"Relative error = {relative_error_1000:.2f}%")

print("\n--- Task 3 ---")

sample_sizes = np.array([100, 1000, 10000, 100000])

estimates = []
errors = []

for n in sample_sizes:
    estimate = monte_carlo_integral(n)
    error = abs(estimate - EXACT)

    estimates.append(estimate)
    errors.append(error)

print(f"{'N':>10} {'Estimate':>15} {'Absolute error':>20}")

for i in range(len(sample_sizes)):
    print(
        f"{sample_sizes[i]:>10} "
        f"{estimates[i]:>15.6f} "
        f"{errors[i]:>20.6f}"
    )

error_100 = errors[0]
error_10000 = errors[2]

if error_10000 != 0:
    error_reduction = error_100 / error_10000
else:
    error_reduction = float("inf")

print()
print(f"Error reduction from N=100 to N=10000: "
      f"{error_reduction:.2f} times")

print("Expected reduction according to 1/sqrt(N): "
      f"{np.sqrt(10000 / 100):.2f} times")

print("\n--- Task 4 ---")

fig, ax = plt.subplots()

ax.plot(sample_sizes, errors, marker="o")

ax.set_xscale("log")

ax.set_xlabel("Number of trials N")
ax.set_ylabel("Absolute error")
ax.set_title("Monte Carlo error convergence - Variant 8")

ax.grid(True)

plt.show()

print("\n--- Task 5 ---")

print("1. The Law of Large Numbers guarantees that the")
print("sample mean approaches the expected value as N increases.")

print()
print("2. The standard error is proportional to sigma / sqrt(N).")
print("Therefore, increasing N by 100 times reduces the")
print("typical error by approximately sqrt(100) = 10 times.")
print("This is why Monte Carlo convergence is proportional")
print("to 1 / sqrt(N), rather than 1 / N.")

# Контрольні питання
#
# 1. Чому оцінка методом Монте-Карло — це, по суті, вибіркове середнє, і який
# результат Лекції 12 гарантує, що вона наближається до точного значення зі
# зростанням N?
#
# Метод Монте-Карло використовує велику кількість випадкових спостережень і
# обчислює їх середнє значення. Для нашого варіанта спочатку генеруються
# випадкові значення x на відрізку від 0 до 4, потім обчислюється sqrt(x), а
# отримане середнє множиться на довжину інтервалу. Наближення до точного
# значення гарантує закон великих чисел. Зі збільшенням кількості випробувань
# вибіркове середнє наближається до математичного сподівання, тому оцінка
# інтеграла наближається до його точного значення.
#
# 2. Чому вчетверо більша кількість випробувань дає лише вдвічі меншу похибку,
# а не вчетверо меншу, — де саме у формулі похибки з'являється корінь?
#
# Стандартна похибка вибіркового середнього визначається формулою:
# SE = sigma / sqrt(N), де sigma — стандартне відхилення, а N — кількість
# випробувань. У формулі присутній квадратний корінь із кількості спостережень.
# Тому якщо збільшити N у 4 рази, то sqrt(N) збільшиться лише у 2 рази.
# Відповідно, типова похибка зменшиться приблизно у 2 рази, а не у 4.
#
# 3. Для варіанта типу "площа": чому частку точок усередині фігури потрібно
# множити саме на площу прямокутника-охоплення?
#
# Для нашого варіанта це питання безпосередньо не застосовується, оскільки
# варіант 8 є задачею на визначений інтеграл, а не на площу фігури. Для задач
# типу "площа" частка випадкових точок, які потрапили всередину фігури,
# показує відношення площі фігури до площі прямокутника, в якому генеруються
# точки. Тому для отримання самої площі потрібно помножити цю частку на площу
# прямокутника-охоплення.
#
# 4. Для варіанта типу "сподівання": чому саме .min(axis=1) чи .max(axis=1)
# дає масив результатів окремо для кожного випробування?
#
# Для нашого варіанта це питання також не застосовується безпосередньо,
# оскільки варіант 8 використовує визначений інтеграл. У задачах зі
# сподіванням матриця містить окремий рядок для кожного випробування. axis=1
# означає виконання операції по кожному рядку окремо. Тому .min(axis=1) або
# .max(axis=1) повертає масив із результатом для кожного випробування. Якщо не
# вказати axis, функція шукатиме мінімум або максимум по всій матриці та
# поверне лише одне число.