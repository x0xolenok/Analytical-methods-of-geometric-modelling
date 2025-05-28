import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Оголошення символів
t = sp.symbols('t')
a, w = sp.symbols('a w', real=True)
P = (w + sp.Abs(t - a) - sp.Abs(t - a - w)) / (2 * w)

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

# Додаємо першу вершину для замикання контуру
vertices.append(vertices[0])
t_points = list(range(len(vertices)))

Rx = 0
Ry = 0
for (x_start, y_start), (x_end, y_end), t_start, t_end in zip(vertices, vertices[1:], t_points, t_points[1:]):
    dx = x_end - x_start
    dy = y_end - y_start
    Rx += dx * P.subs({a: t_start, w: (t_end - t_start)})
    Ry += dy * P.subs({a: t_start, w: (t_end - t_start)})

Rx = sp.simplify(Rx)
Ry = sp.simplify(Ry)

print("x(t) =", Rx)
print("y(t) =", Ry)

fx = sp.lambdify(t, Rx, 'numpy')
fy = sp.lambdify(t, Ry, 'numpy')

tt = np.linspace(t_points[0], t_points[-1], 800)
x_vals = fx(tt)
y_vals = fy(tt)

plt.figure(figsize=(6, 6))
plt.plot(x_vals, y_vals, color='blue', linewidth=2)
plt.axis('equal')
plt.title("Контур літери X")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
