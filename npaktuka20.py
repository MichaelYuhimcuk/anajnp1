from scipy.optimize import linprog

STUDENT = "Mykola Yukhymchuk"
STUDENT_GROUP = "IT-41"
VARIANT = 8

print(f"Student: {STUDENT}")
print(f"Group: {STUDENT_GROUP}")
print(f"Variant: {VARIANT}")

profit_A = 65
profit_B = 75

raw_A = 4
raw_B = 5
raw_limit = 340

time_A = 5
time_B = 3
time_limit = 260

electric_A = 1
electric_B = 1
electric_limit = 90

print("\n--- Task 1 ---")

print("Variables:")
print("x1 - number of product A per week")
print("x2 - number of product B per week")

print("\nObjective:")
print("Maximize P = 65*x1 + 75*x2")

print("\nConstraints:")
print("4*x1 + 5*x2 <= 340  (raw material)")
print("5*x1 + 3*x2 <= 260  (working time)")
print("x1 + x2 <= 90       (electricity)")
print("x1 >= 0, x2 >= 0")

print("\n--- Task 2 ---")
c = [-profit_A, -profit_B]

A_ub = [
    [raw_A, raw_B],
    [time_A, time_B],
    [electric_A, electric_B]
]

b_ub = [
    raw_limit,
    time_limit,
    electric_limit
]

bounds = [
    (0, None),
    (0, None)
]

res = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=bounds,
    method="highs"
)

print("res.x =", res.x)
print("res.fun =", res.fun)
print("Maximum profit =", -res.fun)
print("res.status =", res.status)
print("res.message =", res.message)

if res.status == 0:
    print("Optimization completed successfully.")
else:
    print("Optimization was not successful.")

print("\n--- Task 3 ---")

x1 = res.x[0]
x2 = res.x[1]
profit = -res.fun

print(f"Product A: {x1:.2f} units per week")
print(f"Product B: {x2:.2f} units per week")
print(f"Maximum profit: {profit:.2f} UAH per week")

print("\nInteger solution:")
print("20 units of product A")
print("52 units of product B")
print("Profit = 5200 UAH per week")

print("\n--- Task 4 ---")

print("Slack values:")
print("Raw material:", res.slack[0])
print("Working time:", res.slack[1])
print("Electricity:", res.slack[2])

print("\nResource usage:")

raw_used = raw_A * x1 + raw_B * x2
time_used = time_A * x1 + time_B * x2
electric_used = electric_A * x1 + electric_B * x2

print(f"Raw material: {raw_used:.2f} / {raw_limit}")
print(f"Working time: {time_used:.2f} / {time_limit}")
print(f"Electricity: {electric_used:.2f} / {electric_limit}")

print("\nActive constraints:")
print("- Raw material")
print("- Working time")

print("\nInactive constraint:")
print("- Electricity")

print("\n--- Task 5 ---")

new_raw_limit = raw_limit * 1.15

b_ub_raw = [
    new_raw_limit,
    time_limit,
    electric_limit
]

res_raw = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub_raw,
    bounds=bounds,
    method="highs"
)

print("\n5.1 Increase raw material limit by 15%")
print(f"New raw material limit: {new_raw_limit:.2f}")
print("New solution:", res_raw.x)
print(f"New profit: {-res_raw.fun:.2f} UAH")

print("\nComparison:")
print(f"Old profit: {profit:.2f} UAH")
print(f"New profit: {-res_raw.fun:.2f} UAH")

new_time_limit = time_limit * 1.15

b_ub_time = [
    raw_limit,
    new_time_limit,
    electric_limit
]

res_time = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub_time,
    bounds=bounds,
    method="highs"
)

print("\n5.2 Increase working time limit by 15%")
print(f"New working time limit: {new_time_limit:.2f}")
print("New solution:", res_time.x)
print(f"New profit: {-res_time.fun:.2f} UAH")

print("\nComparison:")
print(f"Old profit: {profit:.2f} UAH")
print(f"New profit: {-res_time.fun:.2f} UAH")

new_electric_limit = electric_limit * 1.50

b_ub_electric = [
    raw_limit,
    time_limit,
    new_electric_limit
]

res_electric = linprog(
    c,
    A_ub=A_ub,
    b_ub=b_ub_electric,
    bounds=bounds,
    method="highs"
)

print("\n5.3 Increase electricity limit by 50%")
print(f"New electricity limit: {new_electric_limit:.2f}")
print("New solution:", res_electric.x)
print(f"New profit: {-res_electric.fun:.2f} UAH")

print("\nComparison:")
print(f"Old profit: {profit:.2f} UAH")
print(f"New profit: {-res_electric.fun:.2f} UAH")

print("\n--- Conclusion ---")

print("The optimal linear programming solution for Poltava is:")
print(f"Product A: {x1:.2f} units")
print(f"Product B: {x2:.2f} units")
print(f"Maximum profit: {profit:.2f} UAH per week")

print("\nThe active constraints are raw material and working time.")
print("Electricity is an inactive constraint with available reserve.")

print("\nIncreasing active resources increases the maximum profit.")
print("Increasing the inactive electricity resource does not change")
print("the optimal solution or the maximum profit.")

# Контрольні питання
#
# 1. Чому linprog мінімізує і як виконати максимізацію?
#
# scipy.optimize.linprog розв'язує задачу мінімізації. Тому для максимізації
# прибутку коефіцієнти цільової функції потрібно помножити на -1. Після
# отримання результату справжній прибуток обчислюється як -res.fun.
#
# 2. Чому >= потрібно множити на -1?
#
# linprog очікує обмеження у формі:
# A_ub @ x <= b_ub
# Тому обмеження виду >= потрібно помножити на -1, щоб перетворити його на
# допустиму форму.
#
# 3. Що таке активне та неактивне обмеження?
#
# Активне обмеження використовується повністю, тому його slack = 0. Воно є
# вузьким місцем виробництва. Неактивне обмеження має запас, тому збільшення
# його ліміту не обов'язково дасть більший прибуток.
#
# 4. Що показав What-if аналіз?
#
# Збільшення активних ресурсів — сировини або робочого часу — дозволило
# збільшити прибуток. Збільшення неактивного ресурсу — електроенергії — не
# змінило ні оптимальний план, ні прибуток. Це підтверджує, що саме активні
# обмеження є вузькими місцями виробництва.