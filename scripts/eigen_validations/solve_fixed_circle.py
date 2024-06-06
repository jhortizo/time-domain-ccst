import numpy as np

from utils import check_eigenvals_convergence


def main():
    geometry_type = "circle"
    force_reprocess = True
    params = {"side_y": 1.0, "side_x": 3.0}
    cst_model = "cst_quad9_rot4"
    constraints_loads = "circle_borders_fixed"

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

    eigsolution = True

    mesh_sizes = np.logspace(0, -1, 3)
    plot_style = "all"

    check_eigenvals_convergence(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        materials,
        eigsolution,
        force_reprocess,
        mesh_sizes,
        plot_style,
    )


if __name__ == "__main__":
    main()
