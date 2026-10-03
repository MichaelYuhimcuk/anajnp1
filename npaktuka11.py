import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

STUDENT_NAME = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"
VARIANT = 8
CITY = "Poltava"
LAM = 0.16

true_mean = 1 / LAM

rng = np.random.default_rng(VARIANT)

print(f"Student: {STUDENT_NAME}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")
print(f"City: {CITY}")
print(f"Lambda: {LAM}")
print(f"Theoretical mean: {true_mean:.3f} min")

print("\n--- Task 1 ---")

sample_sizes = [5, 50, 500, 20000]

for n in sample_sizes:
    sample = rng.exponential(scale=1 / LAM, size=n)
    sample_mean = sample.mean()
    difference = abs(sample_mean - true_mean)

    print(
        f"n = {n:5d} | "
        f"mean = {sample_mean:.4f} | "
        f"absolute difference = {difference:.4f}"
    )

print("\n--- Task 2 ---")

n_repeats = 10000

means_n10 = rng.exponential(
    scale=1 / LAM,
    size=(n_repeats, 10)
).mean(axis=1)

means_n300 = rng.exponential(
    scale=1 / LAM,
    size=(n_repeats, 300)
).mean(axis=1)

print(f"n = 10: mean of sample means = {means_n10.mean():.4f}")
print(f"n = 300: mean of sample means = {means_n300.mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(means_n10, bins=40, density=True)
axes[0].set_title(
    f"Sampling distribution, n=10\nstd={means_n10.std():.3f}"
)
axes[0].set_xlabel("Sample mean")
axes[0].set_ylabel("Density")

axes[1].hist(means_n300, bins=40, density=True)
axes[1].set_title(
    f"Sampling distribution, n=300\nstd={means_n300.std():.3f}"
)
axes[1].set_xlabel("Sample mean")
axes[1].set_ylabel("Density")

plt.tight_layout()
plt.show()

print("\n--- Task 3 ---")

n_values = [10, 30, 100, 300]

theoretical_se = []
measured_std = []

for n in n_values:
    sample_means = rng.exponential(
        scale=1 / LAM,
        size=(n_repeats, n)
    ).mean(axis=1)

    se_theoretical = (1 / LAM) / np.sqrt(n)
    std_measured = sample_means.std()

    theoretical_se.append(se_theoretical)
    measured_std.append(std_measured)

results = pd.DataFrame({
    "n": n_values,
    "Theoretical SE": theoretical_se,
    "Measured std": measured_std
})

print(results.to_string(index=False))

print("\n--- Task 4 ---")

n1 = 25
n2 = 100

se_25 = (1 / LAM) / np.sqrt(n1)
se_100 = (1 / LAM) / np.sqrt(n2)

print(f"SE for n=25:  {se_25:.4f}")
print(f"SE for n=100: {se_100:.4f}")
print(f"SE ratio: {se_25 / se_100:.2f}")

print("\n--- Task 5 ---")

n_small_repeats = 50

means_50 = rng.exponential(
    scale=1 / LAM,
    size=(n_small_repeats, 10)
).mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(means_n10, bins=40, density=True)
axes[0].set_title("n=10, n_repeats=10000")
axes[0].set_xlabel("Sample mean")
axes[0].set_ylabel("Density")

axes[1].hist(means_50, bins=15, density=True)
axes[1].set_title("n=10, n_repeats=50")
axes[1].set_xlabel("Sample mean")
axes[1].set_ylabel("Density")

plt.tight_layout()
plt.show()

# ## Контрольні питання
#
# **1. Чим відрізняється закон великих чисел від центральної граничної теореми?**
#
# Закон великих чисел показує, що при збільшенні розміру вибірки вибіркове середнє x̄ наближається до теоретичного математичного сподівання E[X]. Тобто він відповідає на питання «куди прямує x̄?».
#
# Центральна гранична теорема описує форму та розкид вибіркового розподілу середнього. При достатньо великому n цей розподіл наближається до нормального, а його стандартне відхилення визначається стандартною похибкою.
#
# **2. Чому вихідний розподіл у практиці обрано сильно скошеним?**
#
# Експоненційний розподіл часу очікування є сильно скошеним. Це дозволяє наочно продемонструвати роботу центральної граничної теореми: хоча окремі спостереження не мають нормального розподілу, розподіл їхніх вибіркових середніх при збільшенні n наближається до нормального.
#
# **3. Чому стандартна похибка зменшується пропорційно √n, а не n?**
#
# Стандартна похибка визначається формулою:
#
# SE = σ / √n.
#
# Тому для зменшення похибки у 2 рази потрібно збільшити кількість спостережень у 4 рази. Наприклад, перехід від n = 25 до n = 100 збільшує вибірку в 4 рази, але зменшує стандартну похибку лише в 2 рази. Це показує закон убуваючої віддачі: для подальшого зменшення похибки потрібно збирати дедалі більше даних.