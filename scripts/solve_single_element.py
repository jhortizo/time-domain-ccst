"""
Hotfix (as in machete), to check the solution for a single element. The functions that are already wrapped
in the module will be modified here just to be used for this specific case.

This script is just meant to keep the analysis reproducible, but I expect it not to be a central part of the codebase.
"""

import warnings

import numpy as np
from run_new_model import plot_fields_quad9_rot4
from run_old_model import plot_fields as plot_fields_quad9
from scipy.sparse.linalg import spsolve
from solidspy.assemutil import assembler, loadasem
from solidspy_uels.solidspy_uels import assem_op_cst, cst_quad9

from time_domain_ccst.cst_utils import assem_op_cst_quad9_rot4, cst_quad9_rot4

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def retrieve_solution_single_element(params, cst_model):
    cst_model_functions = {
        "cst_quad9_rot4": (assem_op_cst_quad9_rot4, cst_quad9_rot4),
        "cst_quad9": (assem_op_cst, cst_quad9),
    }
    assem_op, cst_element = cst_model_functions[cst_model]
    omega = 1  # TODO: would I need to modify this a lot?
    mats = np.array(
        [
            [
                1.0,
                0.3,
                1.0,
                1.0,
            ]
        ]
    )

    side = params["side"]
    nodes = np.array(
        [
            [0.0, -side / 2, -side / 2],
            [0.0, side / 2, -side / 2],
            [0.0, side / 2, side / 2],
            [0.0, -side / 2, side / 2],
            [0.0, 0.0, -side / 2],
            [0.0, side / 2, 0.0],
            [0.0, 0.0, side / 2],
            [0.0, -side / 2, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    npts = nodes.shape[0]

    # Elements
    elements = np.array([[0, 4, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]])

    cons = np.zeros((npts, 3), dtype=int)
    # lower border is fixed in y and roll in x
    # left border is fixed in x and roll in y
    lower_border = [0, 1, 4]
    left_border = [0, 7, 3]

    upper_border = [3, 6, 2]

    cons[list(lower_border), 1] = -1
    cons[list(left_border), 0] = -1
    loads = np.zeros((npts, 4))  # empty loads
    loads[:, 0] = np.arange(npts)  # specify nodes

    loads[list(upper_border), 1 + 1] = 1  # force in y direction

    assem_op, bc_array, neq = assem_op(cons, elements)
    stiff_mat, mass_mat = assembler(
        elements, mats, nodes, neq, assem_op, uel=cst_element, sparse=False
    )

    rhs = loadasem(loads, bc_array, neq)
    # Solution
    solution = spsolve(stiff_mat - omega**2 * mass_mat, rhs)

    return bc_array, solution, nodes, elements


def main():
    params = {"side": 2.0, "mesh_size": 1}

    bc_array, solution, nodes, elements = retrieve_solution_single_element(
        params, "cst_quad9_rot4"
    )

    plot_fields_quad9_rot4(bc_array, nodes, elements, solution)

    bc_array, solution, nodes, elements = retrieve_solution_single_element(
        params, "cst_quad9"
    )

    plot_fields_quad9(bc_array, nodes, elements, solution)


if __name__ == "__main__":
    main()
