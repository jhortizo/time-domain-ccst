"""
Time-marching scheme is proposed for classical continuum mechanics, taking as
initial state an eigenvector of the system, and without loads.
"""

import matplotlib.pyplot as plt
import numpy as np
from solidspy.postprocesor import complete_disp

from time_domain_ccst.constants import IMAGES_FOLDER
from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import (
    plot_fields_quad9_rot4,
    plot_oscillatory_movement,
    plot_oscillatory_movement_sample_points_complete_animation,
    plot_oscillatory_movement_singleplot,
)

plt.rcParams["image.cmap"] = "YlGnBu_r"
plt.rcParams["mathtext.fontset"] = "cm"


def prepare_animation_structure(bc_array, nodes, solutions, n_iter_t):
    # get the indices of the bottom line nodes
    lower_border_ids = np.where(nodes[:, 2] == 0)[0]

    # calculate the displacement solution for each time step
    solution_displacements = np.zeros((len(nodes), 2, n_iter_t))
    for i in range(n_iter_t):
        solution_displacements[:, :, i] = complete_disp(
            bc_array[:, :2], nodes, solutions[:, i], ndof_node=2
        )

    # get the displacement solution for the bottom line nodes
    lower_border_y_displacement = solution_displacements[lower_border_ids, 1, :]

    x_values = nodes[lower_border_ids, 1]
    Y_values = lower_border_y_displacement
    return x_values, Y_values.transpose(), solution_displacements


def main():
    geometry_type = "rectangle"
    params = {"side_x": 10.0, "side_y": 1.0, "mesh_size": 0.5}
    force_reprocess = False
    cst_model = "cst_quad9_rot4"
    constraints_loads = "cantilever_support"

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

    # first solve the eigenvalue problem and acquire an eigenstate
    scenario_to_solve = "eigenproblem"

    bc_array, _, eigvecs, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        materials,
        scenario_to_solve="eigenproblem",
        force_reprocess=force_reprocess,
    )
    print("Number of elements:", len(elements))
    n_eigvec = 11
    # plot_fields_quad9_rot4(
    #     bc_array, nodes, elements, eigvecs[:, n_eigvec], instant_show=True
    # )

    # the mesh and constraints are the same, so the exact structure of the eigvecs array
    # can be used as initial state

    initial_state = eigvecs[:, n_eigvec]
    scenario_to_solve = "time-marching"
    n_t_iter = 1000

    dt = 0.01
    custom_str = f"mode_{n_eigvec}_n_t_iter_{n_t_iter}_dt_{dt}"
    bc_array, solutions, nodes, _ = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        materials,
        scenario_to_solve=scenario_to_solve,
        force_reprocess=force_reprocess,
        dt=dt,
        n_t_iter=n_t_iter,
        initial_state=initial_state,
        custom_str=custom_str,
    )

    x_values, Y_values, solution_displacements = prepare_animation_structure(
        bc_array, nodes, solutions, n_t_iter
    )
    ts = np.linspace(0, n_t_iter * dt, n_t_iter)
    # plot_oscillatory_movement(
    #     x_values,
    #     ts,
    #     Y_values,
    #     n_plots=200,
    #     fps=10,
    #     savepath=IMAGES_FOLDER + f"/ccst_fixed_cantilever_{custom_str}_implicit.gif",
    # )

    # plot_oscillatory_movement_singleplot(
    #     x_values,
    #     ts[:2000],
    #     Y_values[:2000, :],
    #     n_plots=20,
    #     xlabel="x",
    #     ylabel="y",
    #     title="Displacement of the bottom line",
    #     savepath=IMAGES_FOLDER + f"/ccst_fixed_cantilever_{custom_str}_implicit.png",
    # )

    plot_oscillatory_movement_sample_points_complete_animation(
        solution_displacements, nodes, ts, custom_str=custom_str, n_plots=200, fps=10
    )


if __name__ == "__main__":
    main()
