import sympy as sp

x, y, t = sp.symbols("x y t")


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

    u_lambdified = sp.lambdify((x, y, t), u_2d, "numpy")

    return u, u_lambdified
