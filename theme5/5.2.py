import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

x, y, z, t, u, v = sp.symbols('x y z t u v', real=True)


def P(a, w, sym=t):
    return (1 / (2 * w)) * (w + sp.Abs(sym - a) - sp.Abs(sym - a - w))


def build_polyline_expr_nd(vertices, t_values=None, sym=t):
    n = len(vertices)
    d = len(vertices[0])
    if t_values is None:
        t_values = list(range(n))

    exprs = [sp.sympify(vertices[0][j]) for j in range(d)]
    # Кусочно-лінійна інтерполяція між точками
    for i in range(1, n):
        dt = t_values[i] - t_values[i - 1]
        for j in range(d):
            d_coord = vertices[i][j] - vertices[i - 1][j]
            exprs[j] += d_coord * P(t_values[i - 1], dt, sym=sym)
    exprs = [sp.simplify(e) for e in exprs]
    return exprs, t_values


def build_ruled_surface(curve1, curve2):
    # Побудова параметричних рівнянь для кожної кривої
    exprs1, t_vals1 = build_polyline_expr_nd(curve1, sym=t)
    exprs2, t_vals2 = build_polyline_expr_nd(curve2, sym=t)

    # Нормалізація параметра: підставляємо t = u*(t_last - t_first)
    u_exprs1 = [sp.simplify(e.subs(t, u * (t_vals1[-1] - t_vals1[0]))) for e in exprs1]
    u_exprs2 = [sp.simplify(e.subs(t, u * (t_vals2[-1] - t_vals2[0]))) for e in exprs2]

    # Лінійна інтерполяція між кривими для параметра v ∈ [0,1]
    x_ruled = sp.simplify(u_exprs1[0] * (1 - v) + u_exprs2[0] * v)
    y_ruled = sp.simplify(u_exprs1[1] * (1 - v) + u_exprs2[1] * v)

    return x_ruled, y_ruled


# Вершини "X"
vertices = [
    (-6, 6),
    (-4, 6),
    (0, 1),
    (4, 6),
    (6, 6),
    (1, 0),
    (6, -6),
    (4, -6),
    (0, -1),
    (-4, -6),
    (-6, -6),
    (-1, 0)
]

# Розбиваємо X на дві криві.
upper_curve = [(-1, 0), (-6, 6), (-4, 6), (0, 1), (4, 6), (6, 6), (1, 0)]
lower_curve = [(1, 0), (6, -6), (4, -6), (0, -1), (-4, -6), (-6, -6), (-1, 0)]

# Побудова лінійчатої поверхні (ruled surface) між верхньою та нижньою кривими
x_ruled, y_ruled = build_ruled_surface(upper_curve, lower_curve)
print("Завдання 5.2")
print("x(u,v) =", sp.pretty(x_ruled))
print("y(u,v) = 1")
print("z(u,v) =", sp.pretty(y_ruled))
# Перетворення символьних виразів у числові функції
f_x = sp.lambdify((u, v), x_ruled, "numpy")
f_y = sp.lambdify((u, v), y_ruled, "numpy")
# Генеруємо сітку параметрів u, v ∈ [0,1]
u_vals = np.linspace(0, 1, 100)
v_vals = np.linspace(0, 1, 100)
# build mesh
U, V = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
X, Y = f_x(U, V), f_y(U, V)

Z = Y
# Fix: Create a properly shaped Z array instead of a scalar
Y = np.ones_like(X)  # Create a 2D array of ones with the same shape as X

# Create a StructuredGrid from the X, Y, Z arrays
grid = pv.StructuredGrid(X, Y, Z)

# Initialize the PyVista plotter
plotter = pv.Plotter()
plotter.add_mesh(grid, cmap='Pastel1', opacity=0.5, show_edges=True)

# Display the interactive window
plotter.show()
