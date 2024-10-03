import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from solidspy.postprocesor import complete_disp

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import (
    plot_fields_classical,
    plot_fields_quad9_rot4,
)


def find_corresponding_eigmodes(classical_eigvecs_u, ccst_eigvecs_u):
    list_closer = []
    for i, ccst_eigvec in enumerate(ccst_eigvecs_u):
        closest_index = np.argmin(
            np.linalg.norm(classical_eigvecs_u - ccst_eigvec, axis=(1, 2))
        )
        list_closer.append((i, closest_index))

    return list_closer


def find_initial_states(
    geometry_type,
    params,
    ccst_model,
    ccst_constraints_loads,
    ccst_materials,
    classical_model,
    classical_constraints_loads,
    classical_materials,
    force_reprocess,
    ccst_n_eigvec,
    plotting=False,
):
    ccst_bc_array, ccst_eigvals, ccst_eigvecs, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        ccst_model,
        ccst_constraints_loads,
        ccst_materials,
        scenario_to_solve="eigenproblem",
        force_reprocess=force_reprocess,
    )

    classical_bc_array, classical_eigvals, classical_eigvecs, nodes, elements = retrieve_solution(
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

    # classical_eigvecs = -1 * classical_eigvecs # THIS DEPENDS ON THE CASE
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
            ccst_eigvecs[:, 10],
            instant_show=True,
        )

        plot_fields_classical(
            classical_bc_array,
            nodes,
            elements,
            classical_eigvecs[:, 11],
            instant_show=True,
        )

    ccst_initial_state = ccst_eigvecs[:, ccst_n_eigvec]
    classical_initial_state = classical_eigvecs[:, classical_n_eigvec]

    return ccst_initial_state, classical_initial_state


def main():
    geometry_type = "rectangle"
    force_reprocess = False
    mesh_size = 0.65

    h = 1.0
    L = 10 * h
    params = {"side_y": h, "side_x": L, "mesh_size": mesh_size}

    # ccst_constraints_loads = "cantilever_support"
    # ccst_constraints_loads = "cantilever_support_disponly"

    # ccst_constraints_loads = "cantilever_double_support"
    # ccst_constraints_loads = "cantilever_double_support_disponly"

    ccst_constraints_loads = "floating"

    ccst_model = "cst_quad9_rot4"
    ccst_materials = np.array(
        [
            [
                1,  # E, young's modulus
                0.29,  # nu, poisson's ratio
                1,  # eta, coupling parameter
                1,  # rho, density
            ]
        ]
    )

    # classical_constraints_loads = "cantilever_support_classical"
    # classical_constraints_loads = "cantilever_double_support_classical"
    classical_constraints_loads = "floating_classical"
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

    a, b = find_initial_states(
        geometry_type,
        params,
        ccst_model,
        ccst_constraints_loads,
        ccst_materials,
        classical_model,
        classical_constraints_loads,
        classical_materials,
        force_reprocess,
        ccst_n_eigvec=10,
        plotting=False,
    )


if __name__ == "__main__":
    main()
