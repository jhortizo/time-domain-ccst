"""
Time-marching scheme is proposed for classical continuum mechanics, taking as
initial state an eigenvector of the system, and without loads.
"""

import numpy as np

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import plot_fields_quad9_rot4


def main():
    geometry_type = "rectangle"
    params = {"side_x": 1.0, "side_y": 10.0, "mesh_size": 0.1}
    force_reprocess = True
    cst_model = "classical"
    constraints_loads = "cantilever_support"

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
    eigsolution = True

    bc_array, eigvals, eigvecs, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        materials,
        eigensolution=eigsolution,
        force_reprocess=force_reprocess,
    )

    n_eigvec = 0
    plot_fields_quad9_rot4(bc_array, nodes, elements, eigvecs[:, n_eigvec], show=True)

    # the mesh and constraints are the same, so the exact structure of the eigvecs array
    # can be used as initial state
    bc_array, solutions, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        materials,
        force_reprocess=force_reprocess,
        dt=0.1,
        n_t_iter=10,
        initial_state=eigvecs[:, 0],
    )

    pass


if __name__ == "__main__":
    main()
