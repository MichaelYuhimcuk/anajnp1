"""СР 9. Метод Монте-Карло: ризик і аналіз чутливості для бюджетної моделі з СР 7.

Результат: кінцевий залишок коштів студента після 9 місяців (вер-тра).
Невизначені входи: допомога батьків, години підробітку, ціна їжі за день, розваги, транспорт.
Виходи: гістограма результату, ймовірність збитку (дефіциту), перцентилі, tornado-діаграма.

Запуск: pip install numpy matplotlib ; python sr9_monte_carlo.py
"""
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(2026)
N = 100_000               # кількість сценаріїв
M = 9                     # місяців

# ---------- детерміновані (відомі) величини ----------
scholarship, hourly_rate, rent = 2000, 100, 4500
days = np.array([30, 31, 30, 31, 31, 28, 31, 30, 31])
fun_base = np.array([900, 700, 600, 2100, 400, 800, 700, 900, 1100])
job_base = np.array([20, 25, 25, 10, 0, 20, 25, 25, 30])
opening = 3000

# ---------- розподіли невизначених параметрів ----------
# кожна функція повертає масив форми (N, M) - окреме значення в кожному місяці кожного сценарію
SAMPLERS = {
    "Допомога батьків (рівномірний 3000-4500)":
        lambda s: rng.uniform(3750 - 750 * s, 3750 + 750 * s, (N, M)),
    "Години підробітку (нормальний, σ=6 год)":
        lambda s: np.clip(rng.normal(job_base, 6 * s, (N, M)), 0, None),
    "Ціна їжі за день (нормальний 120, σ=20)":
        lambda s: rng.normal(120, 20 * s, (N, M)),
    "Розваги (рівномірний 70-150% від плану)":
        lambda s: fun_base * rng.uniform(1 - 0.3 * s, 1 + 0.5 * s, (N, M)),
    "Транспорт (нормальний 600, σ=100)":
        lambda s: rng.normal(600, 100 * s, (N, M)),
}
NOMINAL = {
    "Допомога батьків (рівномірний 3000-4500)": np.full((N, M), 4000.0),
    "Години підробітку (нормальний, σ=6 год)": np.tile(job_base, (N, 1)).astype(float),
    "Ціна їжі за день (нормальний 120, σ=20)": np.full((N, M), 120.0),
    "Розваги (рівномірний 70-150% від плану)": np.tile(fun_base, (N, 1)).astype(float),
    "Транспорт (нормальний 600, σ=100)": np.full((N, M), 600.0),
}
NAMES = list(SAMPLERS)


def model(parents, hours, food, fun, transport):
    """Та сама модель, що в СР 7, векторизована: повертає баланси (N, M)."""
    income = scholarship + parents + hours * hourly_rate
    expenses = rent + food * days + transport + fun
    return opening + np.cumsum(income - expenses, axis=1)


def run(sample_only=None, scale=1.0):
    """Невизначеним робимо лише параметр sample_only (None -> усі); решта номінальні."""
    inp = {}
    for name in NAMES:
        random_it = sample_only is None or name == sample_only
        inp[name] = SAMPLERS[name](scale) if random_it else NOMINAL[name]
    return model(*[inp[n] for n in NAMES])


# ---------- 1. Основна симуляція ----------
bal = run()
final = bal[:, -1]
print(f"Сценаріїв: {N}")
print(f"Кінцевий залишок: середнє {final.mean():.0f}, медіана {np.median(final):.0f}, "
      f"σ {final.std():.0f} грн")
p5, p50, p95 = np.percentile(final, [5, 50, 95])
print(f"5-й / 50-й / 95-й перцентилі: {p5:.0f} / {p50:.0f} / {p95:.0f} грн")
p_final_neg = (final < 0).mean()
p_any_neg = (bal.min(axis=1) < 0).mean()
print(f"Ймовірність від'ємного кінцевого залишку: {p_final_neg:.1%}")
print(f"Ймовірність дефіциту хоча б в один місяць: {p_any_neg:.1%}")
nominal_final = model(*[NOMINAL[n] for n in NAMES])[0, -1]
print(f"Детермінований (номінальний) результат: {nominal_final:.0f} грн")

# ---------- 2. Аналіз чутливості: лише один параметр випадковий ----------
sens = {}
for name in NAMES:
    out = run(sample_only=name)[:, -1]
    sens[name] = out.std()
sens_sorted = dict(sorted(sens.items(), key=lambda kv: kv[1], reverse=True))
print("\nСтандартне відхилення кінцевого залишку, коли випадковий лише один параметр:")
for k, v in sens_sorted.items():
    print(f"  {k:<45} σ = {v:8.0f} грн")
print("Найсильніший вплив:", next(iter(sens_sorted)))

# зміна діапазону одного параметра (масштаб 0.5x, 1x, 2x) -> як змінюється розподіл
top = next(iter(sens_sorted))
print(f"\nМасштаб діапазону параметра '{top}':")
scale_res = {}
for sc in (0.5, 1, 2):
    out = run(sample_only=top, scale=sc)[:, -1]
    scale_res[sc] = out
    print(f"  x{sc}: σ = {out.std():.0f}, P(залишок<0) = {(out < 0).mean():.1%}")

# ---------- 3. Графіки ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

axes[0].hist(final, bins=60, color="steelblue", edgecolor="white")
axes[0].axvline(0, color="red", lw=2, label=f"збиток: {p_final_neg:.1%}")
axes[0].axvline(final.mean(), color="black", ls="--", label=f"середнє {final.mean():.0f}")
axes[0].axvline(p5, color="gray", ls=":", label=f"5-й перцентиль {p5:.0f}")
axes[0].set_xlabel("Кінцевий залишок, грн")
axes[0].set_ylabel("Кількість сценаріїв")
axes[0].set_title("Розподіл результату (Монте-Карло)")
axes[0].legend(frameon=False, fontsize=8)

names = list(sens_sorted)[::-1]
axes[1].barh([n.split(" (")[0] for n in names], [sens_sorted[n] for n in names],
             color="tab:orange")
axes[1].set_xlabel("σ кінцевого залишку, грн (випадковий лише цей параметр)")
axes[1].set_title("Чутливість: tornado")

for sc, out in scale_res.items():
    axes[2].hist(out, bins=60, alpha=0.5, label=f"діапазон x{sc}")
axes[2].axvline(0, color="red")
axes[2].set_xlabel("Кінцевий залишок, грн")
axes[2].set_ylabel("Кількість сценаріїв")
axes[2].set_title(f"Вплив діапазону: {top.split(' (')[0]}")
axes[2].legend(frameon=False)
fig.tight_layout()
fig.savefig("sr9_monte_carlo.png", dpi=150)
print("\nГрафік збережено: sr9_monte_carlo.png")