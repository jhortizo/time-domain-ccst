import numpy as np

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import plot_fields_quad9_rot4


def main():
    geometry_type = "quarter_ring"
    params = {"inner_radius": 1.0, "outer_radius": 3.0, "mesh_size": 0.5}
    force_reprocess = True
    cst_model = "cst_quad9_rot4"
    constraints_loads = "quarter_ring_rollers_axial_load"

    materials = np.array(
        [
            [
                1,  # E, young's modulus
                0.29,  # nu, poisson's ratio
                1,  # eta, coupling parameter
                1,  # rho, density
            ]
        ]
    )

    bc_array, solution, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        materials,
        scenario_to_solve="static",
        force_reprocess=force_reprocess,
    )

    plot_fields_quad9_rot4(bc_array, nodes, elements, solution)


if __name__ == "__main__":
    main()
