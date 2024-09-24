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


def find_initial_states(
    geometry_type,
    params,
    cst_model,
    ccst_constraints_loads,
    ccst_materials,
    classical_model,
    classical_constraints_loads,
    classical_materials,
    force_reprocess,
    ccst_n_eigvec,
    plotting=False,
):
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

    list_closer = find_corresponding_eigmodes(classical_eigvecs_u, ccst_eigvecs_u)
    classical_n_eigvec = list_closer[ccst_n_eigvec][1]

    if plotting:
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

    return ccst_initial_state, classical_initial_state


def get_dynamic_solution(
    geometry_type,
    params,
    model,
    constraints_loads,
    materials,
    initial_state,
    custom_str,
    dt,
    n_t_iter,
    force_reprocess,
):
    scenario_to_solve = "time-marching"
    geometry_type = "rectangle"
    params = {"side_x": 10.0, "side_y": 1.0, "mesh_size": 1.0}

    bc_array, solutions, nodes, _ = retrieve_solution(
        geometry_type,
        params,
        model,
        constraints_loads,
        materials,
        scenario_to_solve=scenario_to_solve,
        force_reprocess=force_reprocess,
        dt=dt,
        n_t_iter=n_t_iter,
        initial_state=initial_state,
        custom_str=custom_str,
    )

    solution_displacements = get_displacements(bc_array, nodes, solutions, n_t_iter)

    return solution_displacements, nodes


def get_displacements(bc_array, nodes, solutions, n_iter_t):
    solution_displacements = np.zeros((len(nodes), 2, n_iter_t))
    for i in range(n_iter_t):
        solution_displacements[:, :, i] = complete_disp(
            bc_array[:, :2], nodes, solutions[:, i], ndof_node=2
        )

    return solution_displacements


def find_corresponding_eigmodes(classical_eigvecs_u, ccst_eigvecs_u):
    list_closer = []
    for i, ccst_eigvec in enumerate(ccst_eigvecs_u):
        closest_index = np.argmin(
            np.linalg.norm(classical_eigvecs_u - ccst_eigvec, axis=(1, 2))
        )
        list_closer.append((i, closest_index))

    return list_closer


def main():
    # -- Different cases run in this script

    ccst_n_eigvec = 0
    static_field_to_plot = "y"
    dt = 0.5

    # ccst_n_eigvec = 2
    # static_field_to_plot = "norm"
    # dt = 0.05

    # ccst_n_eigvec = 3
    # static_field_to_plot = "y"
    # dt = 0.05

    # -- Overall constants
    geometry_type = "rectangle"
    params = {"side_x": 10.0, "side_y": 1.0, "mesh_size": 1.0}
    force_reprocess = False
    plotting = False

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

    n_t_iter = 1000

    # -- Finding initial states
    ccst_initial_state, classical_initial_state = find_initial_states(
        geometry_type,
        params,
        cst_model,
        ccst_constraints_loads,
        ccst_materials,
        classical_model,
        classical_constraints_loads,
        classical_materials,
        force_reprocess,
        ccst_n_eigvec=ccst_n_eigvec,
        plotting=plotting,
    )

    # -- Getting dynamic solutions
    custom_str = f"mode_{ccst_n_eigvec}_n_t_iter_{n_t_iter}_dt_{dt}_eta_{ccst_materials[0, 2]}_ccst"
    ccst_solution_displacements, nodes = get_dynamic_solution(
        geometry_type,
        params,
        cst_model,
        ccst_constraints_loads,
        ccst_materials,
        ccst_initial_state,
        custom_str,
        dt,
        n_t_iter,
        force_reprocess,
    )

    custom_str = f"mode_{ccst_n_eigvec}_n_t_iter_{n_t_iter}_dt_{dt}_eta_{ccst_materials[0, 2]}_classical"
    classical_solution_displacements, nodes = get_dynamic_solution(
        geometry_type,
        params,
        classical_model,
        classical_constraints_loads,
        classical_materials,
        classical_initial_state,
        custom_str,
        dt,
        n_t_iter,
        force_reprocess,
    )

    # -- Fancy plotting
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
