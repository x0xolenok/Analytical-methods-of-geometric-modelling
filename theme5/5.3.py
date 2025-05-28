import numpy as np
import pyvista as pv

# 1) Задаємо 2D-вершини (закритий контур, без повторення першої точки в кінці)
verts2d = np.array([
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
    (-1, 0),
])
n = len(verts2d)

# 2) Вбудовуємо їх у площину z=0
points3d = np.column_stack((verts2d[:, 0], verts2d[:, 1], np.zeros(n)))

# 3) Описуємо один заповнений полігон (face)
#    формат: [n, i0, i1, ..., i{n-1}]
faces = np.hstack([[n, *range(n)]])

poly = pv.PolyData(points3d, faces)

# 4) Тріангуємо контур (отримуємо чистий набір трикутників)
triangulated = poly.triangulate()

# 5) Екструдуємо цей трикутник-королівський набір вздовж осі Z
H = 10.0
# без capping — бо капи ми зробили правильними трикутниками
prism = triangulated.extrude([0, 0, H], capping=False)

# 6) Додаємо «днище» і «кришку» вручну (достатньо просто копіювати триангуляцію)
bottom = triangulated.copy()
top = triangulated.copy()
top.points[:, 2] = H

# З’єднуємо все в один об’єкт
prism_with_caps = prism + bottom + top

# 7) Відображаємо
plotter = pv.Plotter()
plotter.add_mesh(prism_with_caps, color='lightblue', show_edges=True, line_width=1)
plotter.add_axes()
plotter.show()
