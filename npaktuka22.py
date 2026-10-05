import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import linregress

STUDENT = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"
VARIANT = 8

base_temp = 8.5
amplitude = 14
trend_per_year = 0.04
noise_scale = 1.0
city = "Poltava"

np.random.seed(VARIANT)


print(f"Student: {STUDENT}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")
print(f"City: {city}")

n_years = 10

dates = pd.date_range(
    start="2015-01-01",
    periods=n_years * 12,
    freq="MS"
)

month = dates.month

years_elapsed = (
    (dates.year - dates.year[0])
    + (dates.month - 1) / 12
)

seasonal = amplitude * np.cos(
    (month - 7) / 12 * 2 * np.pi
)

trend = trend_per_year * years_elapsed

noise = np.random.normal(
    0,
    noise_scale,
    len(dates)
)

temp = np.round(
    base_temp + trend + seasonal + noise,
    1
)

climate_ts = pd.Series(
    temp,
    index=dates,
    name="temperature"
)


print()
print("--- Generated time series ---")
print(climate_ts.head(12))
print(f"Number of observations: {len(climate_ts)}")
print(f"Start date: {climate_ts.index[0].date()}")
print(f"End date: {climate_ts.index[-1].date()}")

print()
print("--- Task 1 ---")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    climate_ts.index,
    climate_ts.values,
    label="Temperature"
)

ax.set_xlabel("Date")
ax.set_ylabel("Temperature, C")
ax.set_title(
    "Monthly temperature in Poltava, 2015-2024"
)

ax.legend()
ax.grid(True)

plt.show()

print()
print("--- Task 2 ---")

ma3 = climate_ts.rolling(
    window=3,
    center=True
).mean()

ma12_centered = climate_ts.rolling(
    window=12,
    center=True
).mean()

ma12_trailing = climate_ts.rolling(
    window=12
).mean()


fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    climate_ts.index,
    climate_ts.values,
    alpha=0.35,
    label="Original"
)

ax.plot(
    ma3.index,
    ma3.values,
    label="Moving average, window=3"
)

ax.plot(
    ma12_centered.index,
    ma12_centered.values,
    label="Centered moving average, window=12"
)

ax.plot(
    ma12_trailing.index,
    ma12_trailing.values,
    label="Trailing moving average, window=12"
)

ax.set_xlabel("Date")
ax.set_ylabel("Temperature, C")
ax.set_title("Moving averages for different windows")
ax.legend()
ax.grid(True)

plt.show()


print("Window 3 removes part of the random noise,")
print("but seasonal fluctuations are still visible.")

print("Window 12 removes most of the yearly seasonal")
print("variation because 12 months correspond to one")
print("complete seasonal cycle.")

print("The trailing moving average has a visible lag")
print("relative to the centered moving average because")
print("it uses only current and previous observations.")

print()
print("--- Task 3 ---")

decomposition = seasonal_decompose(
    climate_ts,
    model="additive",
    period=12
)

trend_component = decomposition.trend
seasonal_component = decomposition.seasonal
residual_component = decomposition.resid


fig = decomposition.plot()

fig.set_size_inches(12, 8)

plt.show()

print()
print("Trend component:")
print(trend_component.dropna().head())
print("...")
print(trend_component.dropna().tail())

seasonal_by_month = (
    seasonal_component
    .groupby(seasonal_component.index.month)
    .mean()
)

warmest_month = seasonal_by_month.idxmax()
coldest_month = seasonal_by_month.idxmin()

warmest_value = seasonal_by_month.max()
coldest_value = seasonal_by_month.min()

print()
print(
    f"Warmest seasonal month: {warmest_month}"
)

print(
    f"Seasonal value for warmest month: "
    f"{warmest_value:.2f} C"
)

print(
    f"Coldest seasonal month: {coldest_month}"
)

print(
    f"Seasonal value for coldest month: "
    f"{coldest_value:.2f} C"
)

residual_std = residual_component.dropna().std()

print()
print(
    f"Residual standard deviation: "
    f"{residual_std:.2f} C"
)

print(
    f"Original noise scale: "
    f"{noise_scale:.2f} C"
)

print()
print("--- Task 4 ---")

n_ahead = 12

trend_clean = trend_component.dropna()

x = np.arange(len(trend_clean))

fit = linregress(
    x,
    trend_clean.values
)

print(
    f"Trend slope: {fit.slope:.6f} C/month"
)

print(
    f"Trend slope per year: "
    f"{fit.slope * 12:.6f} C/year"
)

print(
    f"Trend p-value: "
    f"{fit.pvalue:.6e}"
)

print(
    f"Trend standard error: "
    f"{fit.stderr:.6f}"
)

x_future = np.arange(
    len(trend_clean),
    len(trend_clean) + n_ahead
)

trend_forecast = (
    fit.intercept
    + fit.slope * x_future
)

seasonal_by_month = (
    decomposition.seasonal
    .groupby(decomposition.seasonal.index.month)
    .mean()
)


future_dates = pd.date_range(
    climate_ts.index[-1]
    + pd.offsets.MonthBegin(1),
    periods=n_ahead,
    freq="MS"
)


seasonal_forecast = (
    seasonal_by_month
    .reindex(future_dates.month)
    .values
)


forecast_values = (
    trend_forecast
    + seasonal_forecast
)


forecast = pd.Series(
    forecast_values,
    index=future_dates,
    name="forecast"
)


print()
print("Forecast for next 12 months:")
print(forecast.round(2))


# Графік історії та прогнозу
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    climate_ts.index,
    climate_ts.values,
    label="Historical temperature"
)

ax.plot(
    forecast.index,
    forecast.values,
    marker="o",
    label="Forecast"
)

ax.axvline(
    climate_ts.index[-1],
    linestyle="--",
    label="Forecast start"
)

ax.set_xlabel("Date")
ax.set_ylabel("Temperature, C")
ax.set_title(
    "Temperature forecast for Poltava"
)

ax.legend()
ax.grid(True)

plt.show()


print()
print("Forecast warning:")
print(
    "The forecast is an extrapolation of the historical "
    "linear trend and seasonal pattern."
)

print(
    "It is not a guarantee because real weather conditions "
    "can change due to climate variability, extreme events, "
    "and other factors that are not included in the model."
)

print(
    "The farther the forecast goes beyond the observed data, "
    "the less reliable the numerical prediction becomes."
)

print()
print("--- Task 5 ---")

alpha = 0.05

slope = fit.slope
stderr = fit.stderr
pvalue = fit.pvalue

ci_low = slope - 1.96 * stderr
ci_high = slope + 1.96 * stderr


print(f"Slope = {slope:.6f} C/month")
print(f"Standard error = {stderr:.6f}")
print(f"p-value = {pvalue:.6e}")

print()
print(
    f"95% confidence interval: "
    f"[{ci_low:.6f}, {ci_high:.6f}] C/month"
)

print()

if pvalue < alpha:
    print(
        "Decision: reject H0 at alpha = 0.05."
    )
    print(
        "There is statistical evidence of a non-zero trend."
    )
else:
    print(
        "Decision: do not reject H0 at alpha = 0.05."
    )
    print(
        "There is not enough statistical evidence of a "
        "non-zero trend."
    )

print()
print("--- Short-window trend comparison ---")

last_2_years = climate_ts.iloc[-24:]

short_decomposition = seasonal_decompose(
    last_2_years,
    model="additive",
    period=12
)

short_trend = short_decomposition.trend.dropna()

x_short = np.arange(len(short_trend))

short_fit = linregress(
    x_short,
    short_trend.values
)

print(
    f"Trend for all 10 years: "
    f"{fit.slope * 12:.6f} C/year"
)

print(
    f"Trend for last 2 years: "
    f"{short_fit.slope * 12:.6f} C/year"
)

print(
    f"P-value for last 2 years: "
    f"{short_fit.pvalue:.6e}"
)

# Контрольні питання
#
# 1. Чому часовий ряд не можна аналізувати тими самими методами, що звичайну
# незалежну вибірку, і що саме в даних це порушує?
#
# Часовий ряд не можна повністю розглядати як звичайну незалежну вибірку,
# тому що його спостереження впорядковані в часі та можуть бути залежними між
# собою. Наприклад, температура в одному місяці пов'язана з температурою в
# сусідніх місяцях. Крім того, часовий ряд може містити тренд і сезонність. У
# нашому випадку температура має річний сезонний цикл і поступове підвищення
# середнього рівня через заданий тренд потепління.
#
# 2. Чому надто мале й надто велике вікно ковзного середнього — обидва
# проблема, і як розмір вікна варто узгоджувати з періодом сезонності?
#
# Надто мале вікно недостатньо згладжує випадковий шум і сезонні коливання,
# тому основна тенденція залишається важкою для спостереження. Надто велике
# вікно, навпаки, може надмірно згладити дані та приховати важливі зміни в
# часовому ряді. Розмір вікна доцільно пов'язувати з періодом сезонності. У
# нашому випадку сезонність має період 12 місяців, тому вікно 12 місяців добре
# прибирає річні сезонні коливання та дозволяє краще побачити довгостроковий
# тренд.
#
# 3. У чому різниця між трейлінговим (center=False) і центрованим
# (center=True) ковзним середнім, і чому центроване недоступне "в реальному
# часі"?
#
# Трейлінгове ковзне середнє використовує поточне та попередні спостереження.
# Тому його можна обчислювати в реальному часі, коли майбутні значення ще
# невідомі. Центроване ковзне середнє використовує значення як до, так і після
# поточної дати. Воно краще показує згладжену тенденцію без такого
# систематичного запізнення, але для його обчислення потрібні майбутні
# спостереження. Тому в реальному часі центроване середнє використовувати
# неможливо для останніх доступних точок.
#
# 4. Чому саме pvalue нахилу тренду, а не тільки знак slope, визначає, чи є
# підстави говорити про статистично підтверджений тренд?
#
# Додатний slope лише показує, що розрахована лінія має зростаючий напрямок.
# Однак це ще не означає, що такий тренд є статистично значущим. Випадковий
# шум також може створити додатний нахил. pvalue дозволяє перевірити нульову
# гіпотезу про те, що істинний нахил дорівнює нулю. Якщо pvalue < 0.05,
# нульову гіпотезу відхиляють і вважають, що є статистичні підстави говорити
# про наявність тренду. Якщо pvalue >= 0.05, доказів недостатньо, навіть якщо
# slope має додатне значення.