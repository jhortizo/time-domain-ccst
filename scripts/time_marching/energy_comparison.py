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
)
from time_domain_ccst.constants import (
    IMAGES_FOLDER
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

    _, solutions, mass_mat, stiff_mat, _, _ = retrieve_solution(
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
        return_matrices=True,
    )

    return solutions, mass_mat, stiff_mat


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

    ccst_n_eigvec = 3
    dts = [0.1, 0.05, 0.01]
    t_final = 100
    n_t_iters = [int(t_final / dt) for dt in dts]

    # -- Overall constants
    geometry_type = "rectangle"
    params = {"side_x": 10.0, "side_y": 1.0, "mesh_size": 1.0}
    force_reprocess = True
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
    energy_ccst = []
    for dt, n_t_iter in zip(dts, n_t_iters):
        custom_str = f"mode_{ccst_n_eigvec}_n_t_iter_{n_t_iter}_dt_{dt}_eta_{ccst_materials[0, 2]}_ccst"
        ccst_solutions, ccst_mass_mat, ccst_stiff_mat = get_dynamic_solution(
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

        # calculate energy
        d_solutions_u = (ccst_solutions[:, 1:] - ccst_solutions[:, :-1]) / dt
        energy = []
        for i in range(n_t_iter - 1):
            energy.append(
                ccst_solutions[:, i].T @ ccst_stiff_mat @ ccst_solutions[:, i]
                + d_solutions_u[:, i].T @ ccst_mass_mat @ d_solutions_u[:, i]
            )
        energy_ccst.append(energy)

    dt = dts[0]
    n_t_iter = n_t_iters[0]
    custom_str = f"mode_{ccst_n_eigvec}_n_t_iter_{n_t_iter}_dt_{dt}_classical"
    cl_solutions, cl_mass_mat, cl_stiff_mat = get_dynamic_solution(
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

    # calculate energy
    energy_classical = []
    d_solutions = (cl_solutions[:, 1:] - cl_solutions[:, :-1]) / dt
    for i in range(n_t_iter - 1):
        energy_classical.append(
            cl_solutions[:, i].T @ cl_stiff_mat @ cl_solutions[:, i]
            + d_solutions[:, i].T @ cl_mass_mat @ d_solutions[:, i]
        )

    # -- Fancy plotting
    ts = [np.linspace(0, t_final, n_t_iter - 1) for n_t_iter in n_t_iters]

    plt.figure(figsize=(8, 3))
    plt.plot(ts[0], energy_classical / energy_classical[0], label=r"Classical, $\Delta t={0}$".format(dts[0]), color="black", linestyle="--")
    colors = ["black", "#757575", "#b5b5b5"]    
    for i, (dt, n_t_iter) in enumerate(zip(dts, n_t_iters)):
        plt.plot(ts[i], energy_ccst[i] / energy_ccst[i][0], label=r"CCST, $\Delta t={0}$".format(dts[i]), color=colors[i], linestyle='-')
    
    plt.xlabel(r"$t$")
    plt.ylabel(r"$E / E_0$")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_FOLDER}/energy_comparison.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
