import sympy as sp
from continuum_mechanics import vector

x, y, t = sp.symbols("x y t")


def manufactured_solution_null_curl_added_oscillations() -> tuple[sp.Matrix, callable]:
    phi = (
        (x - x**2) ** 2
        * (y - y**2) ** 2
        * sp.sin(6 * sp.pi * x)
        * sp.cos(6 * sp.pi * y)
        * sp.cos(sp.pi * t)
    )
    u = vector.grad(phi)
    u_2d = sp.Matrix([u[0], u[1]])

    u_lambdified = sp.lambdify((x, y), u_2d, "numpy")
    curl_u = vector.curl(u)
    curl_lambdified = sp.lambdify((x, y), curl_u[2], "numpy")
    return u, u_lambdified, curl_lambdified
