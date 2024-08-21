import sympy as sp
from continuum_mechanics import vector

from time_domain_ccst.mms.proposed_solutions import manufactured_solution

x, y = sp.symbols("x y")


def check_null_rotational(manufactured_solution: callable) -> None:
    u, _, _ = manufactured_solution()
    curl_u = vector.curl(u)

    sp.plotting.plot3d(curl_u[2], (x, 0, 1), (y, 0, 1))

    assert curl_u[2].subs(x, 0) == 0
    assert curl_u[2].subs(x, 1) == 0
    assert curl_u[2].subs(y, 0) == 0
    assert curl_u[2].subs(y, 1) == 0


def check_boundary_conditions(manufactured_solution: callable) -> None:
    u, _, _ = manufactured_solution()

    u_x = u[0]
    u_y = u[1]

    sp.plotting.plot3d(u_x, (x, 0, 1), (y, 0, 1), title="u_x")
    sp.plotting.plot3d(u_y, (x, 0, 1), (y, 0, 1), title="u_y")

    assert u_x.subs(x, 0) == 0
    assert u_x.subs(x, 1) == 0
    assert u_x.subs(y, 0) == 0
    assert u_x.subs(y, 1) == 0
    assert u_y.subs(x, 0) == 0
    assert u_y.subs(x, 1) == 0
    assert u_y.subs(y, 0) == 0
    assert u_y.subs(y, 1) == 0


if __name__ == "__main__":
    fcn_to_check = manufactured_solution

    # check_null_rotational(fcn_to_check)
    check_boundary_conditions(fcn_to_check)
