"""СР 11 (допоміжний). Зведення кроків очищення для розділу звіту.

Порівнює dataset_raw.csv та dataset_tidy.csv і формує sr11_cleaning_summary.md,
який можна вставити в підсумковий звіт.

Запуск: pip install pandas ; python sr11_cleaning_summary.py
"""
import pandas as pd

raw = pd.read_csv("dataset_raw.csv")
tidy = pd.read_csv("dataset_tidy.csv", parse_dates=["datetime"])

lines = ["# Кроки очищення набору", ""]
lines.append(f"- Оригінал: {raw.shape[0]} рядків x {raw.shape[1]} стовпців; "
             f"tidy: {tidy.shape[0]} x {tidy.shape[1]}.")
lines.append(f"- Дублікати рядків в оригіналі: {raw.duplicated().sum()}; "
             f"дублікати (станція, час) у tidy: "
             f"{tidy.duplicated(subset=['station', 'datetime']).sum()}.")
lines.append("- Стовпці year/month/day/hour об'єднано в `datetime`; службовий `No` видалено.")
lines.append("- Назви приведено до змістовного нижнього регістру (PM2.5 -> pm25, wd -> "
             "wind_direction тощо); `station`, `wind_direction` - категоріальні типи.")
lines.append("- Додано похідні стовпці `season` і `aqi_category`.")
lines.append("")
lines.append("## Пропуски (% від усіх рядків)")
lines.append("")
lines.append("| Стовпець | Пропусків | % |")
lines.append("|---|---|---|")
miss = tidy.isna().sum().sort_values(ascending=False)
for col, n in miss.items():
    lines.append(f"| {col} | {n} | {n / len(tidy) * 100:.1f} |")

lines.append("")
lines.append("## Викиди (правило 1.5·IQR, значення НЕ видалялись)")
lines.append("")
lines.append("| Стовпець | Викидів | % |")
lines.append("|---|---|---|")
for col in ["pm25", "pm10", "temperature", "wind_speed"]:
    s = tidy[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    n = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    lines.append(f"| {col} | {n} | {n / len(s) * 100:.1f} |")

lines += ["", "## Рішення щодо пропусків і викидів (заповніть власними словами)", "",
          "- Пропуски: не імпутовано; при агрегації до добових значень день враховано, "
          "якщо є щонайменше 18 годинних вимірювань.",
          "- Викиди: екстремально високий PM2.5 - реальні дні смогу, а не помилки, тому "
          "залишено; для моделей використано ln(PM2.5)."]

with open("sr11_cleaning_summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print("\nЗбережено: sr11_cleaning_summary.md")