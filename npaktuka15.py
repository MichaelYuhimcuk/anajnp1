import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare

STUDENT = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"
VARIANT = 8
CITY = "Poltava"

print(f"Student: {STUDENT}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")
print(f"City: {CITY}")

np.random.seed(42)

base_temp = 8.5
amplitude = 14

def get_season(month):
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


rows = []

for year in [2021, 2022, 2023, 2024]:
    for month in range(1, 13):

        seasonal = amplitude * np.cos(
            (month - 7) / 12 * 2 * np.pi
        )

        noise = np.random.normal(0, 1.0)

        temperature = round(
            base_temp + seasonal + noise,
            1
        )

        difference = temperature - base_temp

        if difference < -3:
            norm_category = "colder"
        elif difference > 3:
            norm_category = "warmer"
        else:
            norm_category = "normal"

        rows.append({
            "city": CITY,
            "year": year,
            "month": month,
            "temperature": temperature,
            "season": get_season(month),
            "deviation_from_norm": norm_category
        })


climate = pd.DataFrame(rows)

print("\n--- Generated data ---")
print(climate.to_string(index=False))

print("\n--- Task 1 ---")

table = pd.crosstab(
    climate["season"],
    climate["deviation_from_norm"]
)

print("\nContingency table:")
print(table)

print("\nNormalized contingency table:")
table_normalized = pd.crosstab(
    climate["season"],
    climate["deviation_from_norm"],
    normalize="index"
)

print(table_normalized.round(3))

print("\nTask 1 conclusion:")
print(
    "The contingency table shows how often each deviation category "
    "occurs in each season."
)

most_common = table.stack().idxmax()
most_common_count = table.stack().max()

print(
    f"The most frequent combination is "
    f"'{most_common[0]}' and '{most_common[1]}': "
    f"{most_common_count} observations."
)

print(
    "This is consistent with the construction of the dataset because "
    "seasonal temperature changes are directly related to deviations "
    "from the annual mean."
)

print(
    "The normalized table contains proportions instead of raw counts. "
    "The values in every row sum to 1, so it is easier to compare "
    "the distribution of categories between seasons."
)

print("\n--- Task 2 ---")

chi2, p_value, degrees_of_freedom, expected = chi2_contingency(table)

expected_table = pd.DataFrame(
    expected,
    index=table.index,
    columns=table.columns
)

print(f"Chi-square statistic: {chi2:.6f}")
print(f"p-value: {p_value:.6f}")
print(f"Degrees of freedom: {degrees_of_freedom}")

print("\nExpected frequencies:")
print(expected_table.round(3))

alpha = 0.05

if p_value < alpha:
    print("\nConclusion:")
    print("Reject H0.")
    print(
        "There is a statistically significant relationship between "
        "season and deviation from the annual temperature norm."
    )
else:
    print("\nConclusion:")
    print("Do not reject H0.")
    print(
        "There is not enough statistical evidence of a relationship "
        "between season and deviation from the annual temperature norm."
    )

difference_table = abs(table - expected_table)

max_cell = difference_table.stack().idxmax()
max_difference = difference_table.stack().max()

print(
    f"\nThe largest difference between observed and expected "
    f"frequency is in the cell '{max_cell[0]}' + "
    f"'{max_cell[1]}': {max_difference:.3f}."
)

print("\n--- Task 3 ---")

print("Expected frequencies:")
print(expected_table.round(3))

minimum_expected = expected_table.min().min()

print(f"\nMinimum expected frequency: {minimum_expected:.3f}")

if (expected_table < 5).any().any():

    print("Some expected frequencies are less than 5.")

    print("\nCells with expected frequency below 5:")

    for row in expected_table.index:
        for column in expected_table.columns:
            value = expected_table.loc[row, column]

            if value < 5:
                print(
                    f"{row} + {column}: {value:.3f}"
                )

    print(
        "\nCategories with small expected frequencies should be "
        "combined with logically related categories."
    )

    climate["combined_deviation"] = climate[
        "deviation_from_norm"
    ].replace({
        "colder": "not_warmer",
        "normal": "not_warmer"
    })

    combined_table = pd.crosstab(
        climate["season"],
        climate["combined_deviation"]
    )

    print("\nCombined contingency table:")
    print(combined_table)

    chi2_combined, p_combined, dof_combined, expected_combined = (
        chi2_contingency(combined_table)
    )

    expected_combined_table = pd.DataFrame(
        expected_combined,
        index=combined_table.index,
        columns=combined_table.columns
    )

    print("\nExpected frequencies after combining:")
    print(expected_combined_table.round(3))

    print(f"\nChi-square statistic: {chi2_combined:.6f}")
    print(f"p-value: {p_combined:.6f}")
    print(f"Degrees of freedom: {dof_combined}")

    if p_combined < alpha:
        print("Conclusion after combining: reject H0.")
    else:
        print("Conclusion after combining: do not reject H0.")

else:
    print(
        "All expected frequencies are at least 5. "
        "The chi-square test condition is satisfied."
    )

print("\n--- Task 4 ---")

season_counts = climate["season"].value_counts()

season_order = [
    "winter",
    "spring",
    "summer",
    "autumn"
]

season_counts = season_counts.reindex(season_order)

print("Observed season frequencies:")
print(season_counts)

n = len(climate)

expected_season_counts = [
    0.25 * n,
    0.25 * n,
    0.25 * n,
    0.25 * n
]

print("\nExpected frequencies:")
print(expected_season_counts)

season_result = chisquare(
    f_obs=season_counts,
    f_exp=expected_season_counts
)

print(f"\nChi-square statistic: {season_result.statistic:.6f}")
print(f"p-value: {season_result.pvalue:.6f}")

if season_result.pvalue < alpha:
    print("\nConclusion:")
    print("Reject H0.")
    print(
        "The distribution of seasons is statistically significantly "
        "different from the expected 25% for each season."
    )
else:
    print("\nConclusion:")
    print("Do not reject H0.")
    print(
        "The distribution of seasons does not significantly differ "
        "from the expected 25% for each season."
    )

print(
    "\nThis result is expected because every year contains exactly "
    "three months of each of the four seasons."
)

print("\n--- Task 5 ---")

print(
    "H0: the distribution of temperature deviations is "
    "60% normal, 20% colder and 20% warmer."
)

category_order = [
    "normal",
    "colder",
    "warmer"
]

observed_deviation = climate[
    "deviation_from_norm"
].value_counts().reindex(category_order, fill_value=0)

print("\nObserved frequencies:")
print(observed_deviation)

n = len(climate)

expected_deviation = [
    0.60 * n,
    0.20 * n,
    0.20 * n
]

print("\nExpected frequencies:")
print(expected_deviation)

deviation_result = chisquare(
    f_obs=observed_deviation,
    f_exp=expected_deviation
)

print(
    f"\nChi-square statistic: "
    f"{deviation_result.statistic:.6f}"
)

print(
    f"p-value: "
    f"{deviation_result.pvalue:.6f}"
)

if deviation_result.pvalue < alpha:
    print("\nConclusion:")
    print("Reject H0.")
    print(
        "The observed distribution of temperature deviations "
        "differs statistically significantly from the proposed "
        "60% / 20% / 20% distribution."
    )
else:
    print("\nConclusion:")
    print("Do not reject H0.")
    print(
        "There is not enough evidence to conclude that the observed "
        "distribution differs from the proposed 60% / 20% / 20% distribution."
    )

# 1. Чому в статистиці хі-квадрат використовується квадрат відхилення і ділення на очікувану частоту?
# Квадрат відхилення потрібен для того, щоб додатні та від'ємні відхилення не компенсували одне одного.
# Ділення на очікувану частоту показує, наскільки велике відхилення є відносно кількості спостережень,
# яку ми очікуємо. Тому великі відхилення від очікуваних значень дають більший внесок у статистику хі-квадрат.
#
# 2. Як обчислюється очікувана частота клітинки таблиці спряженості?
# Очікувана частота обчислюється за формулою:
# очікувана частота = (сума рядка * сума стовпця) / загальна сума.
# Ця формула показує, скільки спостережень ми очікували б отримати у конкретній клітинці,
# якби дві категоріальні змінні були незалежними.
#
# 3. Чим відрізняється твердження "p-value велике — немає підстав відхилити H0"
# від твердження "p-value велике — доведено, що змінні незалежні"?
# Велике p-value означає лише те, що ми не маємо достатніх статистичних підстав відхилити нульову гіпотезу H0.
# Це не означає, що незалежність змінних була доведена. Тест просто не виявив статистично значущого зв'язку
# на наявних даних.
#
# 4. Що робити, якщо очікувана частота в якійсь клітинці менша за 5 і чому це проблема?
# Якщо очікувана частота менша за 5, наближення розподілом хі-квадрат може бути неточним, тому результат тесту
# може бути ненадійним. У такому випадку можна об'єднати логічно близькі категорії, щоб збільшити очікувані
# частоти, після чого повторити критерій хі-квадрат.