from time_domain_ccst.mms.proposed_solutions import manufactured_solution_1
from continuum_mechanics import vector


def check_null_rotational(manufactured_solution: callable) -> None:
    u, u_fnc = manufactured_solution()

    curl_u = vector.curl(u)

    assert curl_u[0] == 0
    assert curl_u[1] == 0
    assert curl_u[2] == 0


if __name__ == "__main__":
    check_null_rotational(manufactured_solution_1)
