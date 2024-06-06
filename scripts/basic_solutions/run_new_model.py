import numpy as np

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import plot_fields_quad9_rot4


def main():
    geometry_type = "square"
    params = {"side": 1.0, "mesh_size": 0.1}
    force_reprocess = True
    cst_model = "cst_quad9_rot4"
    constraints_loads = "lower_roller_left_roller_upper_force"

    materials = np.array(
        [
            [
                200e9,  # E, young's modulus
                0.29,  # nu, poisson's ratio
                1e4,  # eta, coupling parameter
                7900,  # rho, density
            ]
        ]
    )

    bc_array, solution, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        materials,
        force_reprocess=force_reprocess,
    )

    plot_fields_quad9_rot4(bc_array, nodes, elements, solution)


if __name__ == "__main__":
    main()
