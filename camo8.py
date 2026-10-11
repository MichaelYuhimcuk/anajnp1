"""СР 8. Транспортна задача та задача про призначення (scipy.optimize.linprog).

Запуск: pip install numpy scipy ; python sr8_transport_assignment.py
"""
import numpy as np
from scipy.optimize import linear_sum_assignment, linprog

# =====================================================================
# 1. ТРАНСПОРТНА ЗАДАЧА
# 3 склади (постачальники) -> 4 магазини (споживачі), вартість за одиницю вантажу, грн
# =====================================================================
suppliers = ["Склад A (Київ)", "Склад B (Львів)", "Склад C (Одеса)"]
consumers = ["Магазин 1", "Магазин 2", "Магазин 3", "Магазин 4"]
supply = np.array([120, 150, 80])          # пропозиція, од.
demand = np.array([90, 110, 80, 70])       # попит, од.
cost = np.array([[8, 6, 10, 9],
                 [9, 12, 13, 7],
                 [14, 9, 16, 5]])
m, n = cost.shape
assert supply.sum() == demand.sum(), "задача має бути збалансованою"

# змінні x_ij (m*n), цільова функція: sum c_ij x_ij -> min
c = cost.ravel()
# обмеження за рядками: sum_j x_ij = supply_i; за стовпцями: sum_i x_ij = demand_j
A_eq, b_eq = [], []
for i in range(m):
    row = np.zeros((m, n)); row[i, :] = 1
    A_eq.append(row.ravel()); b_eq.append(supply[i])
for j in range(n):
    col = np.zeros((m, n)); col[:, j] = 1
    A_eq.append(col.ravel()); b_eq.append(demand[j])

res = linprog(c, A_eq=np.array(A_eq), b_eq=b_eq, bounds=(0, None), method="highs")
plan = res.x.reshape(m, n).round(6)
print("=== Транспортна задача ===")
print("Статус:", res.message)
print("План перевезень (од.):")
print(f"{'':<18}" + "".join(f"{c_:>11}" for c_ in consumers) + f"{'Пропозиція':>12}")
for i in range(m):
    print(f"{suppliers[i]:<18}" + "".join(f"{plan[i, j]:>11.0f}" for j in range(n))
          + f"{supply[i]:>12}")
print(f"{'Попит':<18}" + "".join(f"{d:>11}" for d in demand))
print(f"Мінімальна сумарна вартість: {res.fun:.0f} грн")
assert np.allclose(plan.sum(axis=1), supply) and np.allclose(plan.sum(axis=0), demand)
print("Перевірка: усі обсяги пропозиції й попиту виконано.")
print("Інтерпретація: ненульові клітинки - маршрути, що використовуються:")
for i in range(m):
    for j in range(n):
        if plan[i, j] > 1e-9:
            print(f"  {suppliers[i]} -> {consumers[j]}: {plan[i, j]:.0f} од. "
                  f"(тариф {cost[i, j]}, {plan[i, j] * cost[i, j]:.0f} грн)")

# двоїсті оцінки: на скільки зросте вартість при +1 од. попиту магазину
duals = res.eqlin.marginals
print("Двоїсті оцінки попиту (грн за додаткову одиницю):",
      dict(zip(consumers, duals[m:].round(2))))

# =====================================================================
# 2. ЗАДАЧА ПРО ПРИЗНАЧЕННЯ
# 4 виконавці -> 4 завдання, мінімізувати сумарний час (год)
# =====================================================================
workers = ["Олена", "Іван", "Марія", "Петро"]
tasks = ["Аналіз", "Дизайн", "Тести", "Документація"]
time = np.array([[9, 2, 7, 8],
                 [6, 4, 3, 7],
                 [5, 8, 1, 8],
                 [7, 6, 9, 4]])
k = time.shape[0]

# змінні x_ij in [0,1]; кожен виконавець рівно 1 завдання, кожне завдання рівно 1 виконавець
A, b = [], []
for i in range(k):
    row = np.zeros((k, k)); row[i, :] = 1
    A.append(row.ravel()); b.append(1)
for j in range(k):
    col = np.zeros((k, k)); col[:, j] = 1
    A.append(col.ravel()); b.append(1)
res2 = linprog(time.ravel(), A_eq=np.array(A), b_eq=b, bounds=(0, 1), method="highs")
x = res2.x.reshape(k, k).round(6)
print("\n=== Задача про призначення ===")
print("Матриця x (0/1):\n", x.astype(int))
for i in range(k):
    j = int(np.argmax(x[i]))
    print(f"  {workers[i]} -> {tasks[j]} ({time[i, j]} год)")
print(f"Мінімальний сумарний час (linprog): {res2.fun:.0f} год")

# перевірка угорським методом
r_idx, c_idx = linear_sum_assignment(time)
print(f"Перевірка (угорський метод): {time[r_idx, c_idx].sum()} год")
assert abs(res2.fun - time[r_idx, c_idx].sum()) < 1e-6

print("""
=== Чим відрізняються задачі ===
- Транспортна: обсяги x_ij - довільні невід'ємні числа; сума по рядку = пропозиція a_i,
  сума по стовпцю = попит b_j (різні числа, що можуть бути великими); один постачальник
  обслуговує багатьох споживачів частинами.
- Про призначення: усі a_i = b_j = 1, змінні x_ij in {0, 1} (або [0,1], що за цілочисловості
  матриці обмежень дає ті самі 0/1-розв'язки); кожен виконавець отримує РІВНО одне завдання,
  кожне завдання виконує РІВНО один виконавець; матриця квадратна (n x n).
- Тобто задача про призначення - окремий випадок транспортної з одиничними запасами й
  потребами, але з додатковою вимогою неподільності ("один до одного").
""")