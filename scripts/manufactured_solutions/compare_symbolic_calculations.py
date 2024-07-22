from time_domain_ccst.mms.utils import (
    calculate_body_force_fcn_continuum_mechanics,
    calculate_body_force_fcn_manually,
    check_manual_vs_continuum_mechanics,
    manufactured_solution,
)


def compare_symbolic_calculations():
    u, u_fnc = manufactured_solution()

    body_force_fcn, f_cm = calculate_body_force_fcn_continuum_mechanics(u)

    body_force_fcn_manual, f_m = calculate_body_force_fcn_manually(u)

    check_manual_vs_continuum_mechanics(
        f_m, f_cm, body_force_fcn_manual, body_force_fcn
    )


if __name__ == "__main__":
    compare_symbolic_calculations()
