import warnings

import numpy as np
from solidspy.postprocesor import complete_disp
from tqdm import tqdm

from time_domain_ccst.mms.plots import (
    conditional_fields_plotting,
    conditional_loads_plotting,
    convergence_plot,
)
from time_domain_ccst.mms.proposed_solutions import manufactured_solution_3
from time_domain_ccst.mms.utils import (
    calculate_body_force_fcn_continuum_mechanics,
    inverse_complete_disp,
    solve_manufactured_solution,
)

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def run_mms():
    plot_loads = "none"
    plot_field = "last"
    force_reprocess = True

    u, u_fnc, curl_fcn = manufactured_solution_3()

    body_force_fcn, _ = calculate_body_force_fcn_continuum_mechanics(u)

    mesh_sizes = np.logspace(np.log10(1), np.log10(1e-2), num=5)

    l2_errors = []
    n_elements = []
    for mesh_size in tqdm(mesh_sizes, desc="Mesh sizes"):
        bc_array, solution, nodes, elements, rhs, mass_mat = (
            solve_manufactured_solution(
                mesh_size, body_force_fcn, force_reprocess=force_reprocess
            )
        )
        print("Mesh size:", len(elements), " elements")

        u_fem = complete_disp(bc_array, nodes, solution, ndof_node=2)

        # correctly reorder array from (2, 1, nnodes) to (nnodes, 2)
        u_true = u_fnc(nodes[:, 1], nodes[:, 2])
        u_true = np.squeeze(u_true)
        u_true = np.swapaxes(u_true, 0, 1)

        conditional_loads_plotting(
            bc_array, nodes, rhs, elements, mesh_size, mesh_sizes, plot_loads
        )
        conditional_fields_plotting(
            u_fem,
            nodes,
            elements,
            u_true,
            mesh_size,
            mesh_sizes,
            plot_field,
            solution,
            bc_array,
            curl_fcn,
        )

        solution_teo = inverse_complete_disp(
            bc_array, nodes, u_true, len(elements), ndof_node=2
        )
        e = solution_teo - solution
        l2_error = e.T @ mass_mat @ e

        n_elements.append(len(elements))
        l2_errors.append(l2_error)

    convergence_plot(
        mesh_sizes,
        l2_errors,
        error_metric_name="Error L2 Norm",
        filename="mms_convergence_l2norm.png",
    )


if __name__ == "__main__":
    run_mms()
