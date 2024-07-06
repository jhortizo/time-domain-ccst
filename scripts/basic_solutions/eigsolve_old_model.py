import numpy as np

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import plot_fields_quad9


def main():
    geometry_type = "square"
    params = {"side": 1.0, "mesh_size": 0.1}
    force_reprocess = True
    cst_model = "cst_quad9"
    constraints_loads = "borders_fixed"

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
    plot_fields_quad9(bc_array, nodes, elements, eigvecs[:, n_eigvec])


if __name__ == "__main__":
    main()
