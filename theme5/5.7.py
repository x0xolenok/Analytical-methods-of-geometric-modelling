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

gamma_z1 = z  # z >= 0
gamma_z2 = 1 - z  # z <= 3  <=> 3 - z >= 0
# Побудова повної 3D ідентифікаційної функції як перетин області в площині та проміжку по z
gamma_xyz = ir(ir(gamma_shape, gamma_z1), gamma_z2)
gamma_xyz_simpl = sp.simplify(gamma_xyz)

# Отримання символьного рівняння контуру поверхні тіла (ω(x,y,z)=0)
print("\nНеявне рівняння контуру поверхні тіла:")
sp.pprint(gamma_xyz_simpl)
print(" = 0")

# Перетворення символьного виразу у числову функцію
f = sp.lambdify((x, y, z), gamma_xyz_simpl, 'numpy')

# Задаємо сітку точок для (x,y,z)
x_vals = np.linspace(-10, 10, 200)
y_vals = np.linspace(-10, 10, 200)
z_vals = np.linspace(-1, 4, 200)
# Використовуємо параметр indexing='ij', щоб координати відповідали порядку (x,y,z)
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
F = f(X, Y, Z)

# Отримання розмірів сітки
nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)

# Створення StructuredGrid:
# Формуємо масив точок розмірності (N,3)
points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = (nx, ny, nz)

# Додаємо дані функції до сітки (дані мають бути у порядку Fortran)
grid["values"] = F.flatten(order="F")

# Отримання ізоповерхні для рівня 0 (тобто ω(x,y,z)=0)
contours = grid.contour(isosurfaces=[0], scalars="values")

# Візуалізація ізоповерхні за допомогою PyVista
plotter = pv.Plotter()
plotter.add_mesh(contours, color="blue", opacity=0.5, label="ω(x,y,z)=0")
plotter.add_axes(xlabel='x', ylabel='y', zlabel='z')
plotter.add_legend()
plotter.add_title("Поверхня тіла, задана системою обмежень")
plotter.show()