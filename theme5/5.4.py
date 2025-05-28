import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

x, y, z, t, u, v = sp.symbols('x y z t u v', real=True)

def P(a, w, sym=t):
    return (1 / (2 * w)) * (w + sp.Abs(sym - a) - sp.Abs(sym - a - w))


# Булеві операції
def ir(u, v):
    """Перетин (мінімум двох виразів)"""
    return (u + v - sp.Abs(u - v)) / 2

def ur(u, v):
    """Об'єднання (максимум двох виразів)"""
    return (u + v + sp.Abs(u - v)) / 2

def dr(u, v):
    """Різниця (u - v)"""
    return (u - v - sp.Abs(u + v)) / 2

def strip(a, b, c, h):
    """
    Генерує смугу:
    lin = a*x + b*y + c,
    w = h*sqrt(a^2+b^2) - |lin|
    """
    lin = a * x + b * y + c
    w = h * sp.sqrt(a ** 2 + b ** 2) - sp.Abs(lin)
    return lin, w

# --- 1. Вертикальна частина літери Х ---

# x ∈ [0,1]
_, vertical_x = strip(1, 1, 0, 1)

# y ∈ [0,4]
_, vertical_y = strip(0, 1, 0, 3)

# Вертикальний прямокутник
vertical_rect = ir(vertical_x, vertical_y)

# --- 2. Горизонтальна частина літери Г ---

# x ∈ [0,3]
_, horizontal_x = strip(1, -1, 0, 1)

# y ∈ [3,4]
_, horizontal_y = strip(0, 1, 0, 3)

# Горизонтальний прямокутник
horizontal_rect = ir(horizontal_x, horizontal_y)

# --- 3. Форма літери ---
# Об'єднання вертикальної та горизонтальної частин
gamma_shape = ur(vertical_rect, horizontal_rect)
print(sp.simplify(gamma_shape))
# Контур (рівень 0)
f = sp.lambdify((x, y), gamma_shape, 'numpy')

# Сітка точок
x_vals = np.linspace(-10, 10, 500)
y_vals = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

plt.figure(figsize=(6, 6))
plt.contour(X, Y, Z, levels=[0], colors='black')
plt.title("Контур X")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()
