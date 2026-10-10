"""СР 2. Власний набір даних: якість повітря (UCI Beijing Multi-Site Air Quality).

Що робить скрипт:
1. завантажує набір з UCI і зберігає оригінал (dataset_raw.csv);
2. перевіряє придатність набору (рядки, стовпці, типи, пропуски, дублікати);
3. приводить до tidy-форми й зберігає dataset_tidy.csv;
4. автоматично формує dataset_description.md.

Запуск:  pip install pandas requests
         python sr2_air_quality.py
"""
import glob
import io
import os
import zipfile
from datetime import date

import pandas as pd
import requests

URL = ("https://archive.ics.uci.edu/static/public/501/"
       "beijing+multi+site+air+quality+data.zip")
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data"
RAW_DIR = "data_raw"
RAW_CSV = "dataset_raw.csv"
TIDY_CSV = "dataset_tidy.csv"
DESC_MD = "dataset_description.md"


# ---------------------------------------------------------------- 1. завантаження
def download_and_extract():
    """Завантажує zip і розпаковує (включно з вкладеними zip)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    if glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True):
        print("CSV вже є в", RAW_DIR, "- завантаження пропущено")
        return
    print("Завантаження:", URL)
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    zipfile.ZipFile(io.BytesIO(resp.content)).extractall(RAW_DIR)
    # всередині може бути ще один zip
    for inner in glob.glob(os.path.join(RAW_DIR, "**", "*.zip"), recursive=True):
        zipfile.ZipFile(inner).extractall(os.path.dirname(inner))


def load_raw():
    download_and_extract()
    files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "PRSA_Data_*.csv"),
                             recursive=True))
    if not files:
        raise FileNotFoundError("Не знайдено PRSA_Data_*.csv у " + RAW_DIR)
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df.to_csv(RAW_CSV, index=False)  # оригінал не змінюємо далі
    print(f"Оригінал збережено: {RAW_CSV} ({len(files)} файлів-станцій)")
    return df


# ---------------------------------------------------------------- 2. перевірка
def check_suitability(df):
    print("\n=== Первинна перевірка ===")
    print("Форма (рядки, стовпці):", df.shape)
    print("\nТипи:\n", df.dtypes)
    print("\nПерші рядки:\n", df.head())
    print("\nПропуски:\n", df.isna().sum())
    print("\nДублікати:", df.duplicated().sum())
    print("\nОпис числових:\n", df.describe().T)

    numeric = df.select_dtypes("number").columns
    categorical = df.select_dtypes(["object", "category"]).columns
    checks = {
        ">= 200 рядків": len(df) >= 200,
        ">= 6 стовпців": df.shape[1] >= 6,
        "є числова змінна": len(numeric) > 0,
        "є категоріальна змінна": len(categorical) > 0,
        "пропусків у PM2.5 < 30%": df["PM2.5"].isna().mean() < 0.30,
    }
    print("\n=== Чек-ліст придатності ===")
    for name, ok in checks.items():
        print(("[x] " if ok else "[ ] ") + name)


# ---------------------------------------------------------------- 3. tidy
def make_tidy(df):
    t = df.copy()
    t["datetime"] = pd.to_datetime(t[["year", "month", "day", "hour"]])
    t = t.drop(columns=["No", "year", "month", "day", "hour"])
    t = t.rename(columns={
        "PM2.5": "pm25", "PM10": "pm10", "SO2": "so2", "NO2": "no2",
        "CO": "co", "O3": "o3", "TEMP": "temperature", "PRES": "pressure",
        "DEWP": "dew_point", "RAIN": "rain", "wd": "wind_direction",
        "WSPM": "wind_speed",
    })
    t["station"] = t["station"].astype("category")
    t["wind_direction"] = t["wind_direction"].astype("category")

    # пора року за місяцем
    month = t["datetime"].dt.month
    season = pd.Series("winter", index=t.index)
    season[month.isin([3, 4, 5])] = "spring"
    season[month.isin([6, 7, 8])] = "summer"
    season[month.isin([9, 10, 11])] = "autumn"
    t["season"] = pd.Categorical(
        season, categories=["winter", "spring", "summer", "autumn"],
        ordered=True)

    # категорія якості повітря за PM2.5 (межі US EPA, мкг/м3)
    bins = [-0.001, 12, 35.4, 55.4, 150.4, 250.4, float("inf")]
    labels = ["Good", "Moderate", "Unhealthy for sensitive",
              "Unhealthy", "Very unhealthy", "Hazardous"]
    t["aqi_category"] = pd.cut(t["pm25"], bins=bins, labels=labels, ordered=True)

    cols = ["datetime", "station", "season", "pm25", "pm10", "so2", "no2",
            "co", "o3", "temperature", "pressure", "dew_point", "rain",
            "wind_speed", "wind_direction", "aqi_category"]
    t = t[cols].sort_values(["station", "datetime"]).reset_index(drop=True)
    t = t.drop_duplicates(subset=["station", "datetime"])
    t.to_csv(TIDY_CSV, index=False)
    print(f"\nTidy-версію збережено: {TIDY_CSV}, форма {t.shape}")
    return t


# ---------------------------------------------------------------- 4. опис
# назва: (тип змінної, одиниці, шкала, опис)
META = {
    "datetime": ("дата/час", "-", "інтервальна", "момент вимірювання (щогодини)"),
    "station": ("категоріальна", "-", "номінальна", "станція моніторингу в Пекіні"),
    "season": ("категоріальна", "-", "порядкова", "пора року (за місяцем)"),
    "pm25": ("кількісна, неперервна", "мкг/м³", "відносна", "концентрація частинок PM2.5"),
    "pm10": ("кількісна, неперервна", "мкг/м³", "відносна", "концентрація частинок PM10"),
    "so2": ("кількісна, неперервна", "мкг/м³", "відносна", "діоксид сірки"),
    "no2": ("кількісна, неперервна", "мкг/м³", "відносна", "діоксид азоту"),
    "co": ("кількісна, неперервна", "мкг/м³", "відносна", "чадний газ"),
    "o3": ("кількісна, неперервна", "мкг/м³", "відносна", "озон"),
    "temperature": ("кількісна, неперервна", "°C", "інтервальна", "температура повітря"),
    "pressure": ("кількісна, неперервна", "гПа", "відносна", "атмосферний тиск"),
    "dew_point": ("кількісна, неперервна", "°C", "інтервальна", "температура точки роси"),
    "rain": ("кількісна, неперервна", "мм", "відносна", "опади"),
    "wind_speed": ("кількісна, неперервна", "м/с", "відносна", "швидкість вітру"),
    "wind_direction": ("категоріальна", "-", "номінальна", "напрямок вітру (16 румбів)"),
    "aqi_category": ("категоріальна", "-", "порядкова", "категорія якості повітря за PM2.5"),
}


def write_description(raw, tidy):
    lines = [
        "# Опис набору даних",
        "",
        "- **Назва:** Beijing Multi-Site Air-Quality Data",
        f"- **Джерело (посилання):** {SOURCE_PAGE}",
        "- **Ліцензія:** CC BY 4.0 (за інформацією на сторінці набору; перевірте)",
        f"- **Дата вивантаження:** {date.today():%d.%m.%Y}",
        f"- **Оригінал:** {RAW_CSV}, {raw.shape[0]} рядків × {raw.shape[1]} стовпців",
        f"- **Tidy-версія:** {TIDY_CSV}, {tidy.shape[0]} рядків × {tidy.shape[1]} стовпців",
        f"- **Період:** {tidy['datetime'].min():%d.%m.%Y} - {tidy['datetime'].max():%d.%m.%Y}",
        f"- **Станцій:** {tidy['station'].nunique()}",
        "- **Один рядок:** одне годинне вимірювання на одній станції.",
        "- **Питання для дослідження:** сезонні відмінності PM2.5, вплив вітру "
        "й температури на забрудненість, порівняння станцій.",
        "",
        "## Стовпці",
        "",
        "| Назва | Тип змінної | Одиниці | Шкала | Опис | Пропусків |",
        "|---|---|---|---|---|---|",
    ]
    for col in tidy.columns:
        typ, unit, scale, desc = META[col]
        lines.append(f"| {col} | {typ} | {unit} | {scale} | {desc} | "
                     f"{tidy[col].isna().sum()} |")
    lines += [
        "",
        "## Пропуски та особливості",
        "",
        "- Пропуски є в більшості вимірюваних показників (датчики іноді не працюють).",
        "- `aqi_category` і `season` обчислені в коді, у вихідному наборі їх немає.",
        "- Усі години й станції подано окремими рядками (довгий формат).",
    ]
    with open(DESC_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Опис збережено:", DESC_MD)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    raw = load_raw()
    check_suitability(raw)
    tidy = make_tidy(raw)
    write_description(raw, tidy)
    print("\nГотово: dataset_raw.csv, dataset_tidy.csv, dataset_description.md")
