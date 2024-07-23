import sympy as sp
from continuum_mechanics import vector

x, y = sp.symbols("x y")


def manufactured_solution_1() -> tuple[sp.Matrix, callable]:
    u = sp.Matrix(
        [
            x * (1 - x) * y * (1 - y) * sp.sin(sp.pi * x) * sp.cos(sp.pi * y),
            x * (1 - x) * y * (1 - y) * sp.cos(sp.pi * x) * sp.sin(sp.pi * y),
            0,
        ]
    )
    # TODO: for some reason here the u.row_del(2) decided not to work, try and fix it
    u_2d = sp.Matrix(
        [
            x * (1 - x) * y * (1 - y) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y),
            x * (1 - x) * y * (1 - y) * sp.cos(sp.pi * x) * sp.cos(sp.pi * y),
        ]
    )

    # u = sp.Matrix(
    #     [sp.sin(sp.pi * x) * sp.sin(sp.pi * y), sp.sin(sp.pi * y) * sp.sin(sp.pi * x)]
    # )

    u_lambdified = sp.lambdify((x, y), u_2d, "numpy")
    return u, u_lambdified


def manufactured_solution_2() -> tuple[sp.Matrix, callable]:
    phi = (x - x**2) ** 2 * (y - y**2) ** 2
    u = vector.grad(phi)
    u_2d = sp.Matrix([u[0], u[1]])

    u_lambdified = sp.lambdify((x, y), u_2d, "numpy")
    return u, u_lambdified
