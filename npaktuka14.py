import numpy as np
from scipy import stats

STUDENT = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"
VARIANT = 8
CITY = "Poltava"

print(f"Student: {STUDENT}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")
print(f"City: {CITY}")

normal_value = 72

morning = np.array([
    84, 81, 76, 69, 63, 61,
    59, 61, 67, 74, 80, 85
])

months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

alpha = 0.05

print("\n--- Task 1 ---")

print("H0: the mean annual humidity is equal to the reference norm.")
print("H1: the mean annual humidity differs from the reference norm.")

t_stat, p_value = stats.ttest_1samp(
    morning,
    popmean=normal_value
)

print(f"Reference norm: {normal_value}%")
print(f"Sample mean: {morning.mean():.2f}%")
print(f"t-statistic: {t_stat:.6f}")
print(f"p-value: {p_value:.6f}")

if p_value < alpha:
    print("Conclusion: reject H0.")
    print("The mean humidity is statistically significantly different from the reference norm.")
else:
    print("Conclusion: do not reject H0.")
    print("There is no statistically significant difference from the reference norm.")

print("\n--- Task 2 ---")

mean_value = morning.mean()
std_value = morning.std(ddof=1)
n = len(morning)

t_manual = (mean_value - normal_value) / (std_value / np.sqrt(n))

print(f"Mean: {mean_value:.6f}")
print(f"Sample standard deviation: {std_value:.6f}")
print(f"Number of observations: {n}")
print(f"Manual t-statistic: {t_manual:.6f}")
print(f"ttest_1samp t-statistic: {t_stat:.6f}")

if np.isclose(t_manual, t_stat):
    print("The manual calculation matches ttest_1samp.")
else:
    print("The manual calculation does not match ttest_1samp.")

print("\n--- Task 3 ---")

lviv = np.array([
    85, 83, 80, 73, 68, 65,
    63, 65, 70, 77, 82, 86
])

print("Comparison: Poltava vs Lviv")
print("Poltava region: Center")
print("Lviv region: West")

lev_stat_3, lev_p_3 = stats.levene(morning, lviv)

print(f"Levene statistic: {lev_stat_3:.6f}")
print(f"Levene p-value: {lev_p_3:.6f}")

if lev_p_3 < alpha:
    equal_var_3 = False
    print("Variances are statistically different.")
    print("Use Welch t-test: equal_var=False.")
else:
    equal_var_3 = True
    print("There is no statistically significant difference between variances.")
    print("Use standard t-test: equal_var=True.")

t_stat_3, p_value_3 = stats.ttest_ind(
    morning,
    lviv,
    equal_var=equal_var_3
)

print(f"t-statistic: {t_stat_3:.6f}")
print(f"p-value: {p_value_3:.6f}")

if p_value_3 < alpha:
    print("Conclusion: reject H0.")
    print("The mean humidity of the two cities is statistically significantly different.")
else:
    print("Conclusion: do not reject H0.")
    print("There is no statistically significant difference between the mean humidity values.")

print("\n--- Task 4 ---")

kyiv = np.array([
    84, 82, 78, 70, 65, 63,
    61, 63, 68, 75, 81, 85
])

odesa = np.array([
    82, 80, 78, 74, 71, 69,
    67, 68, 72, 76, 80, 83
])

kharkiv = np.array([
    83, 80, 75, 68, 62, 60,
    58, 60, 66, 73, 79, 84
])

dnipro = np.array([
    82, 79, 74, 67, 61, 59,
    57, 59, 65, 72, 78, 83
])

chernihiv = np.array([
    85, 83, 79, 71, 66, 63,
    61, 63, 69, 76, 82, 86
])

ivano_frankivsk = np.array([
    86, 84, 81, 75, 70, 67,
    65, 67, 72, 78, 83, 87
])

sumy = np.array([
    85, 82, 77, 70, 64, 62,
    60, 62, 68, 75, 81, 86
])

uzhhorod = np.array([
    84, 82, 79, 73, 68, 66,
    64, 66, 71, 77, 82, 85
])

cities = {
    "Kyiv": kyiv,
    "Odesa": odesa,
    "Kharkiv": kharkiv,
    "Dnipro": dnipro,
    "Chernihiv": chernihiv,
    "Ivano-Frankivsk": ivano_frankivsk,
    "Sumy": sumy,
    "Uzhhorod": uzhhorod
}

selected_city = None
selected_data = None
lev_stat_4 = None
lev_p_4 = None

for city_name, city_data in cities.items():
    stat, p = stats.levene(morning, city_data)

    if (lev_p_3 < alpha and p >= alpha) or (lev_p_3 >= alpha and p < alpha):
        selected_city = city_name
        selected_data = city_data
        lev_stat_4 = stat
        lev_p_4 = p
        break

if selected_city is None:
    print("No city with opposite Levene result was found.")
else:
    print(f"Selected city: {selected_city}")
    print(f"Levene statistic: {lev_stat_4:.6f}")
    print(f"Levene p-value: {lev_p_4:.6f}")

    if lev_p_4 < alpha:
        equal_var_4 = False
        print("Variances are statistically different.")
        print("Use Welch t-test: equal_var=False.")
    else:
        equal_var_4 = True
        print("There is no statistically significant difference between variances.")
        print("Use standard t-test: equal_var=True.")

    t_stat_4, p_value_4 = stats.ttest_ind(
        morning,
        selected_data,
        equal_var=equal_var_4
    )

    print(f"t-statistic: {t_stat_4:.6f}")
    print(f"p-value: {p_value_4:.6f}")

    if p_value_4 < alpha:
        print("Conclusion: reject H0.")
        print("The mean humidity values are statistically significantly different.")
    else:
        print("Conclusion: do not reject H0.")
        print("There is no statistically significant difference between the mean humidity values.")

print("\n--- Task 5 ---")

subtract_values = np.array([
    3, 3, 4, 5, 6, 7,
    7, 7, 6, 5, 4, 3
])

evening = morning - subtract_values

print("Morning values:")
print(morning)

print("Evening values:")
print(evening)

t_stat_5, p_value_5 = stats.ttest_rel(
    morning,
    evening
)

print(f"t-statistic: {t_stat_5:.6f}")
print(f"p-value: {p_value_5:.6f}")

if p_value_5 < alpha:
    print("Conclusion: reject H0.")
    print("The difference between morning and evening humidity is statistically significant.")
else:
    print("Conclusion: do not reject H0.")
    print("There is no statistically significant difference between morning and evening humidity.")

print("\nWhy paired t-test?")
print("Morning and evening measurements belong to the same months.")
print("Each morning value has a corresponding evening value.")
print("The paired test analyzes the difference within each pair.")
print("This removes the influence of month-specific conditions and reduces variability.")

# 1. Чому для перевірки середнього однієї вибірки використовується саме t-, а не z-розподіл?
# Для одновибіркового t-критерію використовується t-розподіл, тому що стандартне відхилення генеральної сукупності невідоме.
# Воно оцінюється за вибіркою за допомогою s. Через додаткову невизначеність використовується t-розподіл,
# який враховує розмір вибірки та кількість ступенів вільності. У моєму випадку для 12 місячних значень
# кількість ступенів вільності дорівнює 11.
#
# 2. Чому неправильний вибір equal_var у ttest_ind може дати систематично неправильний p-value?
# Параметр equal_var визначає, чи вважаються дисперсії двох вибірок однаковими. Якщо дисперсії насправді різні,
# а встановити equal_var=True, буде використано неправильну оцінку стандартної похибки. Це впливає на t-статистику
# та p-value і може призвести до неправильного статистичного висновку. Тому перед вибором варіанта ttest_ind
# доцільно перевірити рівність дисперсій за допомогою критерію Лівена. Якщо p < 0.05, використовується Welch-варіант
# (equal_var=False), а якщо p >= 0.05 — стандартний варіант (equal_var=True).
#
# 3. Чому парний критерій потужніший за незалежний двовибірковий на тих самих «до/після» даних?
# Парний t-критерій враховує зв'язок між двома вимірюваннями однієї пари. Для кожної пари обчислюється різниця між
# значеннями. Таким чином усувається частина варіації, яка є спільною для обох вимірювань, наприклад вплив
# конкретного місяця. Завдяки цьому зменшується випадкова похибка і тест може краще виявити реальну систематичну
# різницю.
#
# 4. Чи узгоджується висновок критерію Лівена з порівнянням .std() двох вибірок?
# Загалом результат критерію Лівена можна порівнювати з величинами стандартних відхилень. Якщо стандартні
# відхилення двох вибірок помітно відрізняються, це може вказувати на різницю дисперсій. Однак остаточний
# статистичний висновок потрібно робити за p-value критерію Лівена, а не лише за візуальним порівнянням .std().
# Саме критерій Лівена формально перевіряє гіпотезу про рівність дисперсій.