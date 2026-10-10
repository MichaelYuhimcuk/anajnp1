# 1. Повторення

# Список мутабельний: елементи можна змінювати після створення.
temps = [21, 19, 23, 25]
print(temps[0])       # 21
print(temps[-1])      # 25 — останній елемент
print(temps[1:3])     # [19, 23] — правий край зрізу не входить
temps.append(18)
temps[0] = 22

# Словник зберігає пари «ключ → значення».
student = {"name": "Olena", "group": "KI-31", "score": 87}
print(student["score"])          # 87
student["age"] = 19              # додати ключ
print(student.get("email", "-")) # безпечний доступ із запасним значенням
for key, value in student.items():
    print(key, value)

# Кортеж немутабельний, на відміну від списку.
point = (3, 5)
# point[0] = 4  # TypeError: кортеж змінювати не можна
x, y = point                    # розпаковка кортежу
a, b = 1, 2
a, b = b, a                    # обмін значень без тимчасової змінної

# Кортеж можна використовувати як ключ словника, список — ні.
# Множина зберігає лише унікальні елементи.
print(set([1, 2, 2, 3]))        # {1, 2, 3}

# Іменовані аргументи можна передавати в будь-якому порядку.
def greet(name, greeting="Hello", punct="!"):
    return f"{greeting}, {name}{punct}"


print(greet("Ivan"))                       # Hello, Ivan!
print(greet("Ivan", punct="?"))            # Hello, Ivan?
print(greet(greeting="Hi", name="Ivan"))   # Hi, Ivan!

# Параметри зі значеннями за замовчуванням ставлять після обов'язкових.
# Не використовуйте змінюваний об'єкт (наприклад, []) як значення за замовчуванням.

# List і dict comprehension.
squares = [x ** 2 for x in range(5)]
even = [x for x in range(10) if x % 2 == 0]
lengths = {word: len(word) for word in ["ab", "cde"]}
print(squares)  # [0, 1, 4, 9, 16]
print(even)     # [0, 2, 4, 6, 8]
print(lengths)  # {'ab': 2, 'cde': 3}


# 2. Чотири функції (цикли замість sum() і len())

def mean(values):
    """Обчислює середнє списку без sum() і len()."""
    total = 0
    count = 0
    for value in values:
        total += value
        count += 1
    return total / count if count > 0 else None


def filter_by(values, min_value=None, max_value=None):
    """Залишає значення в межах [min_value, max_value]; None означає без межі."""
    return [
        value for value in values
        if (min_value is None or value >= min_value)
        and (max_value is None or value <= max_value)
    ]


def count_items(items):
    """Підраховує, скільки разів зустрічається кожен елемент."""
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def min_max(values):
    """Повертає кортеж (мінімум, максимум) за один прохід."""
    lo = hi = values[0]
    for value in values:
        if value < lo:
            lo = value
        if value > hi:
            hi = value
    return lo, hi


# Приклад використання функцій.
temps = [21, 19, 23, 25, 18, 22]

print(mean(temps))                           # 21.333...
print(filter_by(temps, min_value=21))        # [21, 23, 25, 22]
print(filter_by(temps, max_value=20))        # [19, 18]
print(count_items(["a", "b", "a", "c", "a"]))  # {'a': 3, 'b': 1, 'c': 1}

lo, hi = min_max(temps)                      # розпаковка повернутого кортежу
print(lo, hi)                                # 18 25


# 3. Зв'язок із NumPy та pandas
# mean(temps)                         -> np.mean(arr)
# filter_by(temps, min_value=21)      -> arr[arr >= 21]
# count_items(...)                    -> pd.Series(...).value_counts()
# min_max(temps)                      -> arr.min(), arr.max()


# 4. Питання для самоперевірки перед тестом
# 1. t = (1, 2); t[0] = 5 — TypeError, бо кортеж немутабельний.
# 2. d["x"] без ключа викликає KeyError; d.get("x") повертає None.
# 3. [x for x in range(6) if x % 3 == 0] повертає [0, 3].
# 4. У def f(a=[]): список створюється один раз і зберігається між викликами.
# 5. lo, hi = min_max(values) розпаковує повернутий кортеж у дві змінні.