import warnings

import numpy as np
from solidspy.postprocesor import complete_disp
from tqdm import tqdm

from time_domain_ccst.mms_t.plots import (
    conditional_fields_plotting,
    convergence_plot,
)
from time_domain_ccst.mms_t.proposed_solutions import (
    manufactured_solution_no_oscillations,
)
from time_domain_ccst.mms_t.utils import (
    calculate_body_force_fcn_continuum_mechanics,
    solve_manufactured_solution,
)

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def run_mms():
    plot_field = "all"
    force_reprocess = True

    u, u_fcn = manufactured_solution_no_oscillations()
    custom_string = "_non_null_curl_no_oscillations"

    body_force_fcn, _ = calculate_body_force_fcn_continuum_mechanics(u)

    # mesh_sizes = np.logspace(0, -2, num=5)[1:]
    mesh_sizes = [0.5]
    # dts = [0.01, 0.005, 0.0025, 0.00125, 0.000625][1: ]
    dts = [0.01]
    final_time = 0.5
    nts = [int(final_time / dt) for dt in dts]

    rms_errors = []
    n_elements = []
    for i, mesh_size in tqdm(enumerate(mesh_sizes), desc="Mesh sizes"):
        bc_array, solutions, nodes, elements, rhs = solve_manufactured_solution(
            mesh_size,
            dts[i],
            nts[i],
            body_force_fcn,
            u_fcn,
            force_reprocess=force_reprocess,
            custom_string=custom_string,
        )
        print("Mesh size:", len(elements), " elements")

        time_array = np.arange(0, final_time, dts[i])
        # correctly reorder array from (2, 1, nnodes, times) to (nnodes, 2, times)
        u_trues = u_fcn(nodes[:, 1], nodes[:, 2], time_array[:, np.newaxis])
        u_trues = np.squeeze(u_trues)
        u_trues = np.transpose(u_trues, (2, 0, 1))

        u_fems = np.zeros_like(u_trues)
        for j in range(len(time_array)):
            u_fems[:, :, j] = complete_disp(
                bc_array, nodes, solutions[:, j], ndof_node=2
            )

        assert np.allclose(u_fems[:, :, 0], u_trues[:, :, 0])
        assert np.allclose(u_fems[:, :, 1], u_trues[:, :, 0])

        conditional_fields_plotting(
            u_fems,
            u_trues,
            plot_field,
            mesh_size,
            mesh_sizes,
            n_points=3,
        )

        n_elements.append(len(elements))
        rms_error = np.sqrt(np.mean((u_trues - u_fems) ** 2))
        rms_errors.append(rms_error)

    convergence_plot(
        mesh_sizes,
        rms_errors,
        error_metric_name="Error RMS",
        filename=f"mms_t_convergence_RMS{custom_string}.png",
    )


if __name__ == "__main__":
    run_mms()
