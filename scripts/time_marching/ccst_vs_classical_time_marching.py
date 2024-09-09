"""
Time-marching scheme is proposed for classical continuum mechanics, taking as
initial state an eigenvector of the system, and without loads.
"""

import matplotlib.pyplot as plt
import numpy as np
from solidspy.postprocesor import complete_disp

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import (
    plot_fields_classical,
    plot_fields_quad9_rot4,
    plot_oscillatory_movement_sample_points_complete_animation_vs_classical,
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


def find_corresponding_eigmodes(classical_eigvecs_u, ccst_eigvecs_u):
    list_closer = []
    for i, ccst_eigvec in enumerate(ccst_eigvecs_u):
        closest_index = np.argmin(
            np.linalg.norm(classical_eigvecs_u - ccst_eigvec, axis=(1, 2))
        )
        list_closer.append((i, closest_index))

    return list_closer


def main():
    geometry_type = "rectangle"
    params = {"side_x": 10.0, "side_y": 1.0, "mesh_size": 1.0}
    force_reprocess = False

    ccst_constraints_loads = "cantilever_support"
    cst_model = "cst_quad9_rot4"
    ccst_materials = np.array(
        [
            [
                1,  # E, young's modulus
                0.29,  # nu, poisson's ratio
                0.1,  # eta, coupling parameter
                1,  # rho, density
            ]
        ]
    )

    classical_constraints_loads = "cantilever_support_classical"
    classical_model = "classical_quad9"
    classical_materials = np.array(
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

    ccst_bc_array, _, ccst_eigvecs, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        ccst_constraints_loads,
        ccst_materials,
        scenario_to_solve="eigenproblem",
        force_reprocess=force_reprocess,
    )

    classical_bc_array, _, classical_eigvecs, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        classical_model,
        classical_constraints_loads,
        classical_materials,
        scenario_to_solve="eigenproblem",
        force_reprocess=force_reprocess,
    )

    ccst_eigvecs_u = np.array(
        [
            complete_disp(ccst_bc_array[:, :2], nodes, ccst_eigvecs[:, i], ndof_node=2)
            for i in range(ccst_eigvecs.shape[1])
        ]
    )

    classical_eigvecs = -1 * classical_eigvecs
    classical_eigvecs_u = np.array(
        [
            complete_disp(
                classical_bc_array, nodes, classical_eigvecs[:, i], ndof_node=2
            )
            for i in range(classical_eigvecs.shape[1])
        ]
    )
    print("Number of elements:", len(elements))

    # different cases run in this script
    # ccst_n_eigvec = 0
    # static_field_to_plot = "y"
    # dt = 0.5
    # ccst_n_eigvec = 2
    # static_field_to_plot = "norm"
    # dt = 0.05
    ccst_n_eigvec = 3
    static_field_to_plot = "y"
    dt = 0.05

    list_closer = find_corresponding_eigmodes(classical_eigvecs_u, ccst_eigvecs_u)
    classical_n_eigvec = list_closer[ccst_n_eigvec][1]

    plot_fields_quad9_rot4(
        ccst_bc_array,
        nodes,
        elements,
        ccst_eigvecs[:, ccst_n_eigvec],
        instant_show=True,
    )

    plot_fields_classical(
        classical_bc_array,
        nodes,
        elements,
        classical_eigvecs[:, classical_n_eigvec],
        instant_show=True,
    )

    ccst_initial_state = ccst_eigvecs[:, ccst_n_eigvec]
    classical_initial_state = classical_eigvecs[:, classical_n_eigvec]
    scenario_to_solve = "time-marching"

    n_t_iter = 1000
    custom_str = (
        f"mode_{ccst_n_eigvec}_n_t_iter_{n_t_iter}_dt_{dt}_eta_{ccst_materials[0, 2]}"
    )

    ccst_bc_array, ccst_solutions, nodes, _ = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        ccst_constraints_loads,
        ccst_materials,
        scenario_to_solve=scenario_to_solve,
        force_reprocess=force_reprocess,
        dt=dt,
        n_t_iter=n_t_iter,
        initial_state=ccst_initial_state,
        custom_str=custom_str,
    )

    classical_bc_array, classical_solutions, nodes, _ = retrieve_solution(
        geometry_type,
        params,
        classical_model,
        classical_constraints_loads,
        classical_materials,
        scenario_to_solve=scenario_to_solve,
        force_reprocess=force_reprocess,
        dt=dt,
        n_t_iter=n_t_iter,
        initial_state=classical_initial_state,
        custom_str=custom_str,
    )

    _, _, ccst_solution_displacements = prepare_animation_structure(
        ccst_bc_array, nodes, ccst_solutions, n_t_iter
    )

    _, _, classical_solution_displacements = prepare_animation_structure(
        classical_bc_array, nodes, classical_solutions, n_t_iter
    )

    ts = np.linspace(0, n_t_iter * dt, n_t_iter)

    plot_oscillatory_movement_sample_points_complete_animation_vs_classical(
        ccst_solution_displacements,
        classical_solution_displacements,
        nodes,
        ts,
        custom_str=custom_str,
        n_plots=200,
        n_points=3,
        fps=10,
        static_field_to_plot=static_field_to_plot,
    )


if __name__ == "__main__":
    main()
