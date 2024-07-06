"""
Time-marching scheme is proposed for classical continuum mechanics, taking as
initial state an eigenvector of the system, and without loads.
"""

import numpy as np

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import plot_fields_classical


def main():
    geometry_type = "rectangle"
    params = {"side_x": 10.0, "side_y": 1.0, "mesh_size": 1.0}
    force_reprocess = True
    cst_model = "classical_quad9"
    constraints_loads = "cantilever_support_classical"

    materials = np.array(
        [
            [
                1,  # E, young's modulus
                0.29,  # nu, poisson's ratio
                1,  # rho, density
            ]
        ]
    )

    # first solve the eigenvalue problem and acquire an eigenstate
    scenario_to_solve = "eigenproblem"

    bc_array, eigvals, eigvecs, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        materials,
        scenario_to_solve="eigenproblem",
        force_reprocess=force_reprocess,
    )

    n_eigvec = 5
    plot_fields_classical(bc_array, nodes, elements, eigvecs[:, n_eigvec], instant_show=True)

    # the mesh and constraints are the same, so the exact structure of the eigvecs array
    # can be used as initial state

    scenario_to_solve = "time-marching"
    bc_array, solutions, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        materials,
        scenario_to_solve=scenario_to_solve,
        force_reprocess=force_reprocess,
        dt=0.1,
        n_t_iter=10,
        initial_state=eigvecs[:, 0],
    )

    for i in range(10):
        plot_fields_classical(bc_array, nodes, elements, solutions[i], show=True)

    pass


if __name__ == "__main__":
    main()
