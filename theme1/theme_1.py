# =====================================================
# Завдання 1.1. Інтерполяційний поліном Лагранжа
# Дані: x = [-4,-3,2,3], y = [3, -1, 2, 0]
# Обчислити поліном, знайти значення в точці x_mid = (-3+2)/2 = -0.5
# та побудувати графік.
# =====================================================

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

print("Task 1.1 Lagrange polynomial")
# задаємо символьну змінну
x = sp.symbols('x')
t = sp.symbols('t')
# варіант 15
X1 = [-4,-3,2,3]
Y1 = [3, -1, 2, 0]


def lagrange_poly(x_val, X, Y):
    n = len(X)
    L = 0
    for i in range(n):
        li = 1
        for j in range(n):
            if j != i:
                li *= (x_val - X[j]) / (X[i] - X[j])
        L += Y[i] * li
    return sp.simplify(L)


L_poly = lagrange_poly(x, X1, Y1)
print("Лагранжів поліном:")
sp.pprint(L_poly)

# Обчислення значення в точці середній між 2-им і 3-им вузлами:
x_mid = (-3 + 2) / 2
L_val = L_poly.subs(x, x_mid)
print(f"\nL({x_mid}) = {L_val.evalf():.5f}")

# Побудова графіка:
x_vals = np.linspace(-5, 3, 400)
L_func = sp.lambdify(x, L_poly, "numpy")
y_vals = L_func(x_vals)

plt.figure(figsize=(6, 4))
plt.plot(x_vals, y_vals, 'b-', label='Лагранжів поліном')
plt.plot(X1, Y1, 'ro', label='Вузли')
plt.plot(x_mid, float(L_val), 'ks', markersize=8, label=f'F({x_mid})')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 1.1. Лагранжів поліном")
plt.grid(True)
plt.show()
# =====================================================
# Завдання 1.2. Інтерполяційний поліном Ньютона
# Дані: x = [0.135,  0.876,  1.336, 2.301,  2.851], y = [2.382,  -0.212,  -1.305,  -3.184,  -4.365]
# =====================================================
print("Task 1.2 (N-th order polynomial interpolation):")
X2 = [0.135,  0.876,  1.336, 2.301,  2.851]
Y2 = [2.382,  -0.212,  -1.305,  -3.184,  -4.365]


def divided_differences(X, Y):
    n = len(X)
    D = np.zeros((n, n), dtype=object)
    for i in range(n):
        D[i, 0] = Y[i]
    for j in range(1, n):
        for i in range(j, n):
            D[i, j] = sp.simplify((D[i, j - 1] - D[i - 1, j - 1]) / (X[i] - X[i - j]))
    return D


D = divided_differences(X2, Y2)
C = [D[i, i] for i in range(len(X2))]
print("Таблиця розділених різниць:")
sp.pprint(sp.Matrix(D))

# Будуємо поліном Ньютона:
N_poly = C[0]
prod = 1
for i in range(1, len(C)):
    prod *= (x - X2[i - 1])
    N_poly += C[i] * prod
N_poly = sp.expand(N_poly)
print("\nНьютонів поліном:")
sp.pprint(N_poly)

# Побудова графіка:
x_vals2 = np.linspace(min(X2) - 1, max(X2) + 1, 400)
N_func = sp.lambdify(x, N_poly, "numpy")
y_vals2 = N_func(x_vals2)

plt.figure(figsize=(6, 4))
plt.plot(x_vals2, y_vals2, 'g-', label="Ньютонів поліном")
plt.plot(X2, Y2, 'ro', label="Вузли")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 1.2. Ньютонів поліном")
plt.grid(True)
plt.show()


# =====================================================
# Завдання 1.3. Інтерполяція функції.
# Обираємо функцію f(x)=x^2+4x+cos^2(x+2) на інтервалі [-10,10].
# Використаємо 8 рівномірних точок.
# =====================================================

print("Task 1.3 (Interpolation of a function):")
def f(x):
    return x ** 2 + 4 * x + np.cos(x + 2) ** 2


# Створимо рівномірну сітку з 6 точок:
X3 = np.linspace(-10, 10, 8)
Y3 = f(X3)

# Побудуємо інтерполяційний поліном Лагранжа (символьна версія)
X3_list = list(X3)
Y3_list = list(Y3)
lag_poly_3 = lagrange_poly(x, X3_list, Y3_list)
lag_poly_3 = sp.expand(lag_poly_3)
print("Інтерполяційний поліном (Лагранжа) для f(x)=x^2+4x+cos^2(x+2):")
sp.pprint(lag_poly_3)

# Побудова графіків: функція, інтерполяція, вузли
x_vals3 = np.linspace(-10, 10, 4000)
f_func = np.vectorize(f)
y_true = f_func(x_vals3)
lag_func_3 = sp.lambdify(x, lag_poly_3, "numpy")
y_interp = lag_func_3(x_vals3)

plt.figure(figsize=(6, 4))
plt.plot(x_vals3, y_true, 'b-', label='f(x)=x^2+4x+cos^2(x+2)')
plt.plot(x_vals3, y_interp, 'r--', label='Інтерполяційний поліном')
plt.plot(X3, Y3, 'ko', markersize=5, label='Вузли')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 1.3. Інтерполяція функції")
plt.grid(True)
plt.show()
# =====================================================
# Завдання 1.4. Наближення функції рядом Тейлора.
# f(x)=e^(-x^2)
# Розкласти функцію e^(-x^2) в околі точки x0 = 0 до 8-го порядку.
# Побудувати графік функції та її часткової суми Тейлора.
# =====================================================

print("Task 1.4 (Approximation of a function using Taylor series):")
# визначаємо функцію f(x) = e^(-x^2)
f_expr = sp.exp(-x**2)

# розкладання функції в ряд Тейлора навколо x0 = 0 до 8-го порядку
order = 8
taylor_poly = sp.series(f_expr, x, 0, order + 1).removeO()
taylor_poly = sp.expand(taylor_poly)
print("Ряд Тейлора для f(x)=e^(-x^2) до 8-го порядку:")
sp.pprint(taylor_poly)

# Побудова графіка
# Обираємо інтервал для x, де функція добре апроксимується, наприклад, [-2, 2]
x_vals = np.linspace(-2, 2, 40000)
# Створюємо чисельну функцію для f(x) та ряду Тейлора
f_func = sp.lambdify(x, f_expr, "numpy")
taylor_func = sp.lambdify(x, taylor_poly, "numpy")

plt.figure(figsize=(6, 4))
plt.plot(x_vals, f_func(x_vals), 'b-', label='f(x) = e^(-x^2)')
plt.plot(x_vals, taylor_func(x_vals), 'r--', label='Часткова сума Тейлора (8-й порядок)')
plt.plot(0, f_func(0), 'ko', label='x0=0')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 1.4. Ряд Тейлора для e^(-x^2)")
plt.grid(True)
plt.show()


# =====================================================
# Завдання 1.5. Інтерполяційний кусково-кубічний поліном Ерміта.
# Дані: вузли X1 = [-4,-3,2,3], Y1 = [3,-1,2,0]
# умова: f(x_i)=y_i та f'(x_i)=1 для усіх i.
# Побудуємо кожну кубічну ланку за формулою:
#
# p_i(x) = y[i-1]*( (h+2*(x - x[i-1]))*(x - x[i])**2/h**3 )
#        + 1*( (x - x[i-1])*(x - x[i])**2/h**2 )
#        + 1*( (x - x[i])*(x - x[i-1])**2/h**2 )
#        + y[i]*( (h-2*(x - x[i]))*(x - x[i-1])**2/h**3 )
#
# де h = x[i]-x[i-1].
# =====================================================
print("Task 1.5 (Interpolation of a function using Hermite polynomials):")
def hermite_segment(x_val, p1, p2):
    x1, y1, g1 = p1
    x2, y2, g2 = p2
    h = x2 - x1
    term1 = y1 * (h + 2 * (x_val - x1)) * (x_val - x2) ** 2 / h ** 3
    term2 = g1 * (x_val - x1) * (x_val - x2) ** 2 / h ** 2
    term3 = g2 * (x_val - x2) * (x_val - x1) ** 2 / h ** 2
    term4 = y2 * (h - 2 * (x_val - x2)) * (x_val - x1) ** 2 / h ** 3
    return sp.simplify(term1 + term2 + term3 + term4)


# Для наших вузлів; похідна в усіх точках = 1.
g_val = 1
nodes = list(zip(X1, Y1, [g_val] * len(X1)))
# Створимо список поліномів для сегментів між вузлами:
segments = []
for i in range(1, len(nodes)):
    seg = hermite_segment(x, nodes[i - 1], nodes[i])
    segments.append(seg)
    print(f"Сегмент між x={nodes[i - 1][0]} і x={nodes[i][0]}:")
    sp.pprint(seg)

# Створимо кускову функцію (Piecewise)
hermite_piecewise = sp.Piecewise(
    (segments[0], (x < nodes[1][0])),
    (segments[1], (x >= nodes[1][0]) & (x < nodes[2][0])),
    (segments[2], (x >= nodes[2][0]))
)
print("\nКускова функція Ерміта:")
sp.pprint(hermite_piecewise)

# Побудова графіка:
x_vals5 = np.linspace(-5, 4, 4000)
hermite_func = sp.lambdify(x, hermite_piecewise, "numpy")
plt.figure(figsize=(6, 4))
plt.plot(x_vals5, hermite_func(x_vals5), 'm-', label="Поліном Ерміта")
plt.plot(X1, Y1, 'ko', markersize=8, label="Вузли")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 1.5. Кусково-кубічний поліном Ерміта")
plt.grid(True)
plt.show()
# =====================================================
# Завдання 1.6. Кубічні сплайни.
# Дані: ті ж вузли X1 = [-4,-3,2,3], Y1 = [3,-1,2,0].
# Побудуємо два сплайни:
#   (a) "затиснутий" – з заданими першими похідними (clamped): f'(x0)=f'(xn)=1,
#   (b) натуральний – з нульовими другими похідними: f''(x0)=f''(xn)=0.
#
# =====================================================

print("Task 1.6 (Interpolation of a function using cubic splines):")

X1_arr = np.array(X1, dtype=float)
Y1_arr = np.array(Y1, dtype=float)


def compute_cubic_spline(x, y, bc_type='natural'):
    n = len(x) - 1
    h = np.diff(x)

    #  Встановлення трикутної системи
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    # Заповнення діагональних записів
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

        # Права частина
        b[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    if bc_type == 'natural':
        # (b) Натуральний сплайн – з f''(x0)=f''(xn)=0
        A[0, 0] = 1.0
        A[n, n] = 1.0
        b[0] = 0.0
        b[n] = 0.0
    elif bc_type == 'clamped':
        # (a) Кубічний сплайн із затисненням (clamped) – задаємо граничні умови (f'(x0)=1, f'(xn)=1)
        A[0, 0] = 2 * h[0]
        A[0, 1] = h[0]
        A[n, n - 1] = h[n - 1]
        A[n, n] = 2 * h[n - 1]

        # Права частина для похідних = 1
        b[0] = 3 * ((y[1] - y[0]) / h[0] - 1)
        b[n] = 3 * (1 - (y[n] - y[n - 1]) / h[n - 1])
    elif bc_type == 'closed':
        # (c) Замкнений сплайн - періодичні граничні умови
        # Для періодичного сплайну: s0 = sn, s'0 = s'n, s''0 = s''n

        # Перший рядок: s0 = sn
        A[0, 0] = 1
        A[0, n] = -1
        b[0] = 0

        # Останній рядок: зв'язок між першою та останньою ділянками
        A[-1, 0] = -2 * h[1]
        A[-1, 1] = -h[1]
        A[-1, -1 - 1] = -h[-1]
        A[-1, -1] = -2 * h[-1]
        b[-1] = 3 * (((y[-1] - y[-1 - 1]) / h[-1]) - ((y[1] - y[0]) / h[1]))
    # Розв'язання системи
    c = np.linalg.solve(A, b)

    # Обчислення коефіцієнтів a, b, d
    a = y[:-1]
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c[:-1], d, x[:-1]


def evaluate_spline(x_eval, x_knots, coeffs):
    a, b, c, d, xi = coeffs
    y_eval = np.zeros_like(x_eval)

    for i in range(len(x_eval)):
        # Шукаємо відповідний інтервал
        idx = np.searchsorted(x_knots, x_eval[i]) - 1
        if idx < 0:
            idx = 0
        if idx >= len(xi):
            idx = len(xi) - 1

        # Обчислити значення сплайна
        dx = x_eval[i] - xi[idx]
        y_eval[i] = a[idx] + b[idx] * dx + c[idx] * dx ** 2 + d[idx] * dx ** 3

    return y_eval


# (a) Кубічний сплайн із затисненням (clamped) – задаємо граничні умови (f'(x0)=1, f'(xn)=1)
coeffs_clamped = compute_cubic_spline(X1_arr, Y1_arr, bc_type='clamped')

# (b) Натуральний сплайн – з f''(x0)=f''(xn)=0
coeffs_natural = compute_cubic_spline(X1_arr, Y1_arr, bc_type='natural')

x_vals6 = np.linspace(min(X1) - 1, max(X1) + 1, 400)
y_clamped = evaluate_spline(x_vals6, X1_arr, coeffs_clamped)
y_natural = evaluate_spline(x_vals6, X1_arr, coeffs_natural)

plt.figure(figsize=(6, 4))
plt.plot(x_vals6, y_clamped, 'b-', label='Кубічний сплайн (clamped)')
plt.plot(x_vals6, y_natural, 'g--', label='Натуральний сплайн')
plt.plot(X1_arr, Y1_arr, 'ro', markersize=8, label='Вузли')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 1.6. Кубічні сплайни")
plt.grid(True)
plt.show()

def symbolic_spline_expression(coeffs, knots, name="Spline", symbol=x):
    a, b, c, d, xi = coeffs

    pieces = []

    for i in range(len(xi)):
        expr = (
            a[i] +
            b[i] * (symbol - xi[i]) +
            c[i] * (symbol - xi[i])**2 +
            d[i] * (symbol - xi[i])**3
        )
        cond = sp.And(symbol >= xi[i], symbol <= knots[i + 1])
        pieces.append((expr, cond))

    spline_piecewise = sp.Piecewise(*pieces)
    print(f"\nСимвольне представлення сплайна: {name}")
    sp.pprint(spline_piecewise)

symbolic_spline_expression(coeffs_clamped, X1_arr, name="Clamped Spline")
symbolic_spline_expression(coeffs_natural, X1_arr, name="Natural Spline")

# =====================================================
# Завдання 1.7. Кубічні параметричні сплайни.
# Дані: точки: [(-2, 3), (1, 2), (1, -4), (-4, -5)]
# Щоб отримати замкнену криву, повторимо першу точку.
# =====================================================
print("Task 1.7 (Interpolation of a function using cubic splines with closed boundary conditions):")
# Масиви точок (x,y):
points_1_7 = [(-2, 3), (1, 2), (1, -4), (-4, -5)]
# Щоб крива була замкненою, додамо першу точку в кінець.
points_closed = points_1_7 + [points_1_7[0]]
t_vals = np.linspace(0, len(points_closed) - 1, len(points_closed))  # параметр t

# Розділяємо x та y
t_arr = np.array(t_vals, dtype=float)
x_points = np.array([p[0] for p in points_closed], dtype=float)
y_points = np.array([p[1] for p in points_closed], dtype=float)

# Використання власної функції для побудови параметричних сплайнів
coeffs_x = compute_cubic_spline(t_arr, x_points, bc_type='closed')
coeffs_y = compute_cubic_spline(t_arr, y_points, bc_type='closed')

# Обчислюємо точки кривої
t_dense = np.linspace(t_arr[0], t_arr[-1], 400)
x_dense = evaluate_spline(t_dense, t_arr, coeffs_x)
y_dense = evaluate_spline(t_dense, t_arr, coeffs_y)

plt.figure(figsize=(6, 6))
plt.plot(x_dense, y_dense, 'b-', label='Параметричний сплайн')
# також намалюємо вхідну ламану (полігон)
plt.plot(x_points, y_points, 'ro--', label='Вхідна ламана')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 1.7. Кубічний параметричний сплайн (замкнута крива)")
plt.axis('equal')
plt.grid(True)
plt.show()

symbolic_spline_expression(coeffs_x, t_arr, name="px(t)",symbol=t)
symbolic_spline_expression(coeffs_y, t_arr, name="py(t)",symbol=t)

# =====================================================
# Завдання 1.8. Поліноміальна апроксимація методом найменших квадратів.
# Дані: використаємо точки з завдання 1.5: X1 = [-4,-3,2,3], Y1 = [3,-1,2,0]
# Обчислити апроксимуючу пряму (лінійна апроксимація) та квадратичний поліном.
# =====================================================
print("Task 1.8 (Interpolation of a function using the least-squares polynomial approximation):")
# Для лінійної апроксимації: шукаємо a0, a1 так, що φ(x)=a0+a1*x мінімізує суму квадратів похибок.
A_lin = np.vstack([np.ones(len(X1_arr)), X1_arr]).T
coeff_lin, residuals_lin, rank, s = np.linalg.lstsq(A_lin, Y1_arr, rcond=None)
print("Коефіцієнти апроксимуючої прямої: a0, a1 =", coeff_lin)

# Для квадратичної апроксимації: φ(x)=a0+a1*x+a2*x^2
A_quad = np.vstack([np.ones(len(X1_arr)), X1_arr, X1_arr ** 2]).T
coeff_quad, residuals_quad, rank, s = np.linalg.lstsq(A_quad, Y1_arr, rcond=None)
print("Коефіцієнти квадратичного полінома: a0, a1, a2 =", coeff_quad)

# Створимо функції
phi_lin = lambda x: coeff_lin[0] + coeff_lin[1] * x
phi_quad = lambda x: coeff_quad[0] + coeff_quad[1] * x + coeff_quad[2] * x ** 2

x_vals8 = np.linspace(min(X1_arr) - 1, max(X1_arr) + 1, 400)
y_lin = phi_lin(x_vals8)
y_quad = phi_quad(x_vals8)

plt.figure(figsize=(6, 4))
plt.plot(x_vals8, y_lin, 'b-', label="Лінійна апроксимація")
plt.plot(x_vals8, y_quad, 'g--', label="Квадратичний поліном")
plt.plot(X1_arr, Y1_arr, 'ro', markersize=8, label="Вузли")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 1.8. Поліноміальна апроксимація (Лінійна та квадратична)")
plt.grid(True)
plt.show()

# Обчислимо відхилення (корінь середньоквадратичної похибки)
delta_lin = np.sqrt(np.sum((phi_lin(X1_arr) - Y1_arr) ** 2))
delta_quad = np.sqrt(np.sum((phi_quad(X1_arr) - Y1_arr) ** 2))
print(f"Відхил апроксимуючої прямої: {delta_lin:.3f}")
print(f"Відхил квадратичного полінома: {delta_quad:.3f}")
# =====================================================
# Завдання 1.9. Апроксимація методом найменших квадратів з використанням функцій 1, x, sin(x), e^(-x)
# Дані:
# Абсциси fіксовано: x = [0,1,2,3,4,5]
# y = [-1.12 -1.41 -2.84 -3.57 -4.06 -5.04]
# =====================================================
print("Task 1.9 (Interpolation of a function using the least-squares polynomial approximation):")
X9 = np.array([0, 1, 2, 3, 4, 5], dtype=float)
Y9 = np.array([-1.12, -1.41, -2.84, -3.57, -4.06, -5.04], dtype=float)

# Створюємо матрицю базисних функцій: 1, x, sin(x), e^(-x)
F1 = np.ones_like(X9)
F2 = X9
F3 = np.sin(X9)
F4 = np.exp(-X9)
A_mat = np.vstack([F1, F2, F3, F4]).T

# Розв'язуємо систему методом найменших квадратів
coeff_ls, residuals, rank, s = np.linalg.lstsq(A_mat, Y9, rcond=None)
print("Коефіцієнти апроксимації (a0, a1, a2, a3):", coeff_ls)


# Побудова апроксимаційного виразу:
# f_approx(x) = a0 + a1*x + a2*sin(x) + a3*e^(-x)
def f_approx(x):
    return coeff_ls[0] + coeff_ls[1] * x + coeff_ls[2] * np.sin(x) + coeff_ls[3] * np.exp(-x)


x_vals9 = np.linspace(min(X9) - 0.5, max(X9) + 0.5, 400)
y_approx = f_approx(x_vals9)

plt.figure(figsize=(6, 4))
plt.plot(x_vals9, y_approx, 'b-', label="Апроксимуюча крива")
plt.plot(X9, Y9, 'ro', markersize=8, label="Дані")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Завдання 1.9. Метод найменших квадратів")
plt.grid(True)
plt.show()