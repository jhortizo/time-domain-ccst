import warnings

import numpy as np
from solidspy.postprocesor import complete_disp
from tqdm import tqdm

from time_domain_ccst.mms.plots import (
    conditional_fields_plotting,
    conditional_loads_plotting,
    convergence_plot,
)
from time_domain_ccst.mms_t.proposed_solutions import (
    manufactured_solution_null_curl_added_oscillations,
)
from time_domain_ccst.mms_t.utils import (
    calculate_body_force_fcn_continuum_mechanics,
    solve_manufactured_solution,
)

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def run_mms():
    plot_loads = "none"
    plot_field = "last"
    force_reprocess = False

    # u, u_fnc, curl_fcn = manufactured_solution_added_oscillations()
    # custom_string = "_non_null_curl_added_oscillation_6"

    # u, u_fnc, curl_fcn = manufactured_solution_null_curl()
    # custom_string = "_null_curl"

    u, u_fnc, curl_fcn = manufactured_solution_null_curl_added_oscillations()
    custom_string = "_null_curl_added_oscillations"

    # u, u_fnc, curl_fcn = manufactured_solution_no_oscillations()
    # custom_string = "_non_null_curl_no_oscillations"

    body_force_fcn, _ = calculate_body_force_fcn_continuum_mechanics(u)

    mesh_sizes = np.logspace(0, -2, num=5)
    dts = [0.01, 0.005, 0.0025, 0.00125, 0.000625]
    final_time = 1.0
    nts = [int(final_time / dt) for dt in dts]

    rms_errors = []
    n_elements = []
    for i, mesh_size in tqdm(enumerate(mesh_sizes), desc="Mesh sizes"):
        bc_array, solutions, nodes, elements, rhs = solve_manufactured_solution(
            mesh_size,
            dts[i],
            nts[i],
            body_force_fcn,
            u_fnc,
            force_reprocess=force_reprocess,
            custom_string=custom_string,
        )
        print("Mesh size:", len(elements), " elements")

        time_array = np.arange(0, final_time, dts[i])
        # correctly reorder array from (2, 1, nnodes) to (nnodes, 2)
        u_true = u_fnc(nodes[:, 1], nodes[:, 2], time_array)
        u_true = np.squeeze(u_true)
        u_true = np.swapaxes(u_true, 0, 1)

        u_fem = complete_disp(bc_array, nodes, solutions, ndof_node=2)

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
            solutions,
            bc_array,
            curl_fcn,
        )

        n_elements.append(len(elements))
        rms_error = np.sqrt(np.mean((u_true - u_fem) ** 2))
        rms_errors.append(rms_error)

    convergence_plot(
        mesh_sizes,
        rms_errors,
        error_metric_name="Error RMS",
        filename=f"mms_t_convergence_RMS{custom_string}.png",
    )


if __name__ == "__main__":
    run_mms()
