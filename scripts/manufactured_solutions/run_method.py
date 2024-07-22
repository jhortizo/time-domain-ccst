import warnings

import numpy as np
from solidspy.postprocesor import complete_disp
from tqdm import tqdm

from time_domain_ccst.mms.plots import (
    conditional_fields_plotting,
    conditional_loads_plotting,
    convergence_plot,
)
from time_domain_ccst.mms.utils import (
    calculate_body_force_fcn_continuum_mechanics,
    manufactured_solution,
    solve_manufactured_solution,
)

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def run_mms():
    plot_loads = "all"
    plot_field = "last"

    u, u_fnc = manufactured_solution()

    body_force_fcn, _ = calculate_body_force_fcn_continuum_mechanics(u)

    mesh_sizes = np.logspace(np.log10(1), np.log10(1e-2), num=5)
    mesh_sizes = mesh_sizes[:-1]  # manually removing the last one

    rmses = []
    max_errors = []
    n_elements = []
    for mesh_size in tqdm(mesh_sizes, desc="Mesh sizes"):
        bc_array, solution, nodes, elements, rhs = solve_manufactured_solution(
            mesh_size, body_force_fcn, force_reprocess=True
        )
        print("Mesh size:", len(elements), " elements")

        u_fem = complete_disp(bc_array, nodes, solution, ndof_node=2)
        # some other dev code to check the loads

        conditional_loads_plotting(
            bc_array, nodes, rhs, elements, mesh_size, mesh_sizes, plot_loads
        )

        # correctly reorder array from (2, 1, nnodes) to (nnodes, 2)
        u_true = u_fnc(nodes[:, 1], nodes[:, 2])
        u_true = np.squeeze(u_true)
        u_true = np.swapaxes(u_true, 0, 1)

        conditional_fields_plotting(
            u_fem, nodes, elements, u_true, mesh_size, mesh_sizes, plot_field
        )

        rmse = np.sqrt(np.mean((u_true - u_fem) ** 2))
        max_error = np.max(np.abs(u_true - u_fem))

        n_elements.append(len(elements))
        rmses.append(rmse)
        max_errors.append(max_error)

    convergence_plot(rmses, max_errors, n_elements)
