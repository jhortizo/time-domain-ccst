from continuum_mechanics import vector
from time_domain_ccst.mms.proposed_solutions import manufactured_solution_3


def check_null_rotational(manufactured_solution: callable) -> None:
    import sympy as sp
    x, y = sp.symbols("x y")
    u, _ = manufactured_solution()

    curl_u = vector.curl(u)

    assert curl_u[2].subs(x, 0) == 0
    assert curl_u[2].subs(x, 1) == 0
    assert curl_u[2].subs(y, 0) == 0
    assert curl_u[2].subs(y, 1) == 0



def check_boundary_conditions(manufactured_solution: callable) -> None:
    import sympy as sp
    x, y = sp.symbols("x y")
    u, _ = manufactured_solution()

    u_x = u[0]
    u_y = u[1]

    assert u_x.subs(x, 0) == 0
    assert u_x.subs(x, 1) == 0
    assert u_x.subs(y, 0) == 0
    assert u_x.subs(y, 1) == 0
    assert u_y.subs(x, 0) == 0
    assert u_y.subs(x, 1) == 0
    assert u_y.subs(y, 0) == 0
    assert u_y.subs(y, 1) == 0


if __name__ == "__main__":
    
    fcn_to_check = manufactured_solution_3
    
    check_null_rotational(fcn_to_check)
    check_boundary_conditions(fcn_to_check)
