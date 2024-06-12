import numpy as np

from utils import check_eigenvals_convergence


def main():
    geometry_type = "rectangle"
    force_reprocess = False
    params = {"side_y": 1.0, "side_x": 3.0}
    cst_model = "cst_quad9_rot4"
    constraints_loads = "floating"

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

    mesh_sizes = np.logspace(0, -0.75, 9)
    plot_style = "last"

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
