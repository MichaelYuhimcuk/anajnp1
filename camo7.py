"""СР 7. Балансова розрахункова модель: місячний бюджет студента на навчальний рік.

Процес: щомісяця надходять доходи (стипендія, підробіток, допомога), витрачаються кошти
на категорії, залишок переноситься на наступний місяць.

Запуск: pip install matplotlib ; python sr7_balance_model.py
"""
import matplotlib.pyplot as plt

# ---------- 1. Вхідні величини (грн) ----------
MONTHS = ["Вер", "Жов", "Лис", "Гру", "Січ", "Лют", "Бер", "Кві", "Тра"]
scholarship = 2000          # стипендія щомісяця
parents = 4000              # допомога щомісяця
job_hours = [20, 25, 25, 10, 0, 20, 25, 25, 30]  # годин підробітку за місяць
hourly_rate = 100           # грн за годину
rent = 4500
food_per_day = 120
days = [30, 31, 30, 31, 31, 28, 31, 30, 31]
transport = 600
fun = [900, 700, 600, 2100, 400, 800, 700, 900, 1100]
opening_balance = 3000      # залишок на початок вересня


# ---------- 2. Формули ----------
def income(m):
    """Дохід за місяць m: I_m = S + P + h_m * r."""
    return scholarship + parents + job_hours[m] * hourly_rate


def expenses(m):
    """Витрати за місяць m: E_m = rent + food_per_day * d_m + transport + fun_m."""
    return rent + food_per_day * days[m] + transport + fun[m]


def simulate(opening=opening_balance):
    """Баланс: B_m = B_{m-1} + I_m - E_m, B_0 = opening."""
    balance = opening
    rows = []
    for m in range(len(MONTHS)):
        i, e = income(m), expenses(m)
        balance = balance + i - e
        rows.append((MONTHS[m], i, e, i - e, balance))
    return rows


rows = simulate()
print(f"{'Міс':<5}{'Дохід':>8}{'Витрати':>9}{'Сальдо':>8}{'Залишок':>9}")
for name, i, e, s, b in rows:
    print(f"{name:<5}{i:>8}{e:>9}{s:>8}{b:>9}")

# ---------- 3. Перевірка моделі на конкретних числах ----------
# ручний розрахунок вересня: дохід 2000+4000+20*100=8000; витрати 4500+120*30+600+900=9600
assert income(0) == 8000 and expenses(0) == 9600
assert rows[0][4] == 3000 + 8000 - 9600                         # 1400
# сума балансу: кінцевий залишок = початковий + всі доходи - всі витрати
total_i = sum(r[1] for r in rows)
total_e = sum(r[2] for r in rows)
assert rows[-1][4] == opening_balance + total_i - total_e
print(f"\nПеревірки пройдено. Усього доходів {total_i}, витрат {total_e}, "
      f"кінцевий залишок {rows[-1][4]} грн.")
neg = [r[0] for r in rows if r[4] < 0]
print("Місяці з від'ємним залишком (дефіцит):", neg or "немає")

# ---------- 4. Чутливість: як зміниться кінцевий залишок ----------
base = rows[-1][4]
extra_hours_effect = 5 * len(MONTHS) * hourly_rate        # +5 год щомісяця
food_effect = -10 * sum(days)                              # їжа +10 грн/день
print(f"\n+5 год підробітку щомісяця: {base + extra_hours_effect} грн "
      f"({extra_hours_effect:+} грн)")
print(f"Їжа +10 грн/день: {base + food_effect} грн ({food_effect:+} грн)")

# ---------- 5. Графік ----------
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(MONTHS, [r[3] for r in rows],
       color=["tab:green" if r[3] >= 0 else "tab:red" for r in rows],
       label="сальдо місяця")
ax.plot(MONTHS, [r[4] for r in rows], "o-", color="black", label="залишок на кінець місяця")
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("грн")
ax.set_title("Місячний баланс бюджету студента")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("sr7_balance.png", dpi=150)

print("""
=== Класифікація моделі ===
- Детермінована: за тих самих вхідних даних завжди той самий результат, випадкових величин
  немає (витрати й доходи вважаємо відомими).
- Динамічна (дискретного часу): стан (залишок) змінюється від місяця до місяця за
  рекурентним співвідношенням B_m = B_{m-1} + I_m - E_m.
- Розрахункова (аналітична, балансова): результат обчислюється за явними формулами;
  оптимізації чи імітації немає.
- Дискретна, лінійна за параметрами: залишок лінійно залежить від вхідних величин.
- Обмеження: реальні витрати випадкові (стохастичні); модель не враховує інфляцію.
""")
