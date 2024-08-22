import sympy as sp
from continuum_mechanics import vector

x, y = sp.symbols("x y")


def manufactured_solution_added_oscillations() -> tuple[sp.Matrix, callable]:
    u = sp.Matrix(
        [
            (x - x**2) ** 2
            * (y - y**2) ** 2
            * sp.sin(6 * sp.pi * x)
            * sp.cos(6 * sp.pi * y),
            (x - x**2) ** 2
            * (y - y**2) ** 2
            * sp.cos(6 * sp.pi * x)
            * sp.sin(6 * sp.pi * y),
            0,
        ]
    )
    # TODO: for some reason here the u.row_del(2) decided not to work, try and fix it
    u_2d = sp.Matrix(
        [
            (x - x**2) ** 2
            * (y - y**2) ** 2
            * sp.sin(6 * sp.pi * x)
            * sp.cos(6 * sp.pi * y),
            (x - x**2) ** 2
            * (y - y**2) ** 2
            * sp.cos(6 * sp.pi * x)
            * sp.sin(6 * sp.pi * y),
        ]
    )

    u_lambdified = sp.lambdify((x, y), u_2d, "numpy")

    curl_u = vector.curl(u)
    curl_lambdified = sp.lambdify((x, y), curl_u[2], "numpy")
    return u, u_lambdified, curl_lambdified


def manufactured_solution_no_oscillations() -> tuple[sp.Matrix, callable]:
    u = sp.Matrix(
        [
            (x - x**2) ** 2 * (y - y**2) ** 2,
            (x - x**2) ** 2 * (y - y**2) ** 2,
            0,
        ]
    )
    # TODO: for some reason here the u.row_del(2) decided not to work, try and fix it
    u_2d = sp.Matrix(
        [
            (x - x**2) ** 2 * (y - y**2) ** 2,
            (x - x**2) ** 2 * (y - y**2) ** 2,
        ]
    )

    u_lambdified = sp.lambdify((x, y), u_2d, "numpy")

    curl_u = vector.curl(u)
    curl_lambdified = sp.lambdify((x, y), curl_u[2], "numpy")
    return u, u_lambdified, curl_lambdified


def manufactured_solution_null_curl() -> tuple[sp.Matrix, callable]:
    phi = (x - x**2) ** 2 * (y - y**2) ** 2
    u = vector.grad(phi)
    u_2d = sp.Matrix([u[0], u[1]])

    u_lambdified = sp.lambdify((x, y), u_2d, "numpy")
    curl_u = vector.curl(u)
    curl_lambdified = sp.lambdify((x, y), curl_u[2], "numpy")
    return u, u_lambdified, curl_lambdified


def manufactured_solution_null_curl_added_oscillations() -> tuple[sp.Matrix, callable]:
    phi = (x - x**2) ** 2 * (y - y**2) ** 2 * sp.sin(6 * sp.pi * x) * sp.cos(6 * sp.pi * y)
    u = vector.grad(phi)
    u_2d = sp.Matrix([u[0], u[1]])

    u_lambdified = sp.lambdify((x, y), u_2d, "numpy")
    curl_u = vector.curl(u)
    curl_lambdified = sp.lambdify((x, y), curl_u[2], "numpy")
    return u, u_lambdified, curl_lambdified