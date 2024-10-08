"""
Solve for wave propagation in classical mechanics in the given domain.
"""

from typing import Literal

import meshio
import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import inv, spsolve
from solidspy.assemutil import DME, assembler, loadasem
from solidspy_uels.solidspy_uels import assem_op_cst, cst_quad9, elast_quad9
from tqdm import tqdm

from .constants import SOLUTION_TYPES
from .constraints_loads_creators import SYSTEMS
from .cst_utils import (
    assem_op_cst_quad9_rot4,
    cst_quad9_rot4,
    decouple_global_matrices,
    get_variables_eqs,
    inverse_complete_disp,
)
from .gmesher import create_mesh
from .utils import (
    check_solution_files_exists,
    generate_solution_filenames,
    load_solutions,
    postprocess_eigsolution,
    save_eigensolution_files,
    save_solution_files,
)

cst_model_functions = {
    "cst_quad9_rot4": (assem_op_cst_quad9_rot4, cst_quad9_rot4),
    "cst_quad9": (assem_op_cst, cst_quad9),
    "classical_quad9": (DME, elast_quad9),
}


def _load_mesh(
    mesh_file: str, cons_loads_fcn: callable, cons_loads_fcn_params: dict
) -> tuple[np.array, np.array, np.array, np.array]:
    mesh = meshio.read(mesh_file)

    points = mesh.points
    cells = mesh.cells
    quad9 = cells["quad9"]
    npts = points.shape[0]
    nels = quad9.shape[0]

    nodes = np.zeros((npts, 3))
    nodes[:, 1:] = points[:, 0:2]

    # Elements
    elements = np.zeros((nels, 3 + 9), dtype=int)
    elements[:, 0] = range(nels)
    elements[:, 1] = 4
    elements[:, 3:] = (
        quad9  # the first 3 cols correspond to material params and elements params
    )
    # the remaining are the nodes ids

    line3 = cells["line3"]
    cell_data = mesh.cell_data
    cons, loads = cons_loads_fcn(line3, cell_data, npts, **cons_loads_fcn_params)

    return cons, elements, nodes, loads


def _compute_solution(
    geometry_type: str,
    geometry_mesh_params: dict,
    cons_loads_fcn_params: dict,
    files_dict: dict,
    cst_model: str,
    cons_loads_fcn: callable,
    materials: np.ndarray,
    scenario_to_solve: Literal["static", "eigenproblem", "time-marching"],
    dt: float | None,
    n_t_iters: int | None,
    initial_state: np.ndarray | dict | None,
    return_matrices: bool,
):
    assem_op, cst_element = cst_model_functions[cst_model]

    # TODO: I need to refactor this later, and to use it in the single case run in independent file
    if geometry_type == "single_element":
        side = geometry_mesh_params["side"]
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

        # Cantilever with support case is hardcoded (TODO: should not be)
        cons = np.zeros((npts, 3), dtype=int)
        left_border = [0, 7, 3]

        cons[list(left_border), :] = -1
        loads = np.zeros((npts, 4))  # empty loads
        loads[:, 0] = np.arange(npts)  # specify nodes

    else:
        create_mesh(geometry_type, geometry_mesh_params, files_dict["mesh"])

        cons, elements, nodes, loads = _load_mesh(
            files_dict["mesh"], cons_loads_fcn, cons_loads_fcn_params
        )

    # Assembly
    can_be_sparse = not (scenario_to_solve == "eigenproblem")
    assem_op, bc_array, neq = assem_op(cons, elements)
    stiff_mat, mass_mat = assembler(
        elements, materials, nodes, neq, assem_op, uel=cst_element, sparse=can_be_sparse
    )

    if scenario_to_solve == "static":
        # static solution does not take into acount the mass matrix
        # for freq. solution, do spsolve(stiff_mat - omega**2 * mass_mat, rhs)
        rhs = loadasem(loads, bc_array, neq)
        # fix nodal loads by weighting them with the mass matrix
        # TODO: only works if density is constant and equal to 1
        rhs = mass_mat @ rhs
        solution = spsolve(stiff_mat, rhs)
        save_solution_files(bc_array, solution, files_dict)
        return bc_array, solution, nodes, elements

    elif scenario_to_solve == "eigenproblem":
        eigvals, eigvecs = eig(stiff_mat, b=mass_mat)
        eigvals, eigvecs = postprocess_eigsolution(eigvals, eigvecs)
        save_eigensolution_files(bc_array, eigvals, eigvecs, files_dict)
        return bc_array, eigvals, eigvecs, nodes, elements

    elif scenario_to_solve == "time-marching":
        rhs = loadasem(loads, bc_array, neq)
        solutions = np.zeros((neq, n_t_iters))
        if type(initial_state) == np.ndarray:
            solutions[:, 0] = initial_state  # assume constant behavior in first steps
            solutions[:, 1] = initial_state
        elif isinstance(initial_state, dict):
            u_x_0 = initial_state["u_x"](nodes[:, 1], nodes[:, 2])
            u_y_0 = initial_state["u_y"](nodes[:, 1], nodes[:, 2])

            if cst_model == "cst_quad9_rot4":
                w_0 = initial_state["w"](nodes[:, 1], nodes[:, 2])
                s_0 = initial_state["s"](nodes[:, 1], nodes[:, 2])

                u_0 = np.zeros((len(nodes), 3))
                u_0[:, 0] = u_x_0
                u_0[:, 1] = u_y_0
                u_0[:, 2] = w_0

                initial_solution = inverse_complete_disp(
                    bc_array, nodes, u_0, len(elements), cst_model, ndof_node=3
                )

                # and the skew-symmetric part of the force-stress tensor
                eqs_s = assem_op[:, 22]
                eqs_s = eqs_s[eqs_s >= 0]
                # get the rows of assem_op where col 22 is not -1
                id_elements_s = np.where(assem_op[:, 22] >= 0)[0]
                id_nodes_s = elements[id_elements_s, 11]
                initial_solution[eqs_s] = s_0[id_nodes_s]

            if cst_model == "classical_quad9":
                u_0 = np.zeros((len(nodes), 2))
                u_0[:, 0] = u_x_0
                u_0[:, 1] = u_y_0
                initial_solution = inverse_complete_disp(
                    bc_array, nodes, u_0, len(elements), cst_model, ndof_node=2
                )

            solutions[:, 0] = initial_solution
            solutions[:, 1] = initial_solution

        if cst_model == "classical_quad9":
            for i in tqdm(range(1, n_t_iters - 1), desc="iterations"):
                A = mass_mat + dt**2 * stiff_mat
                b = (
                    dt**2 * rhs
                    + 2 * mass_mat @ solutions[:, i]
                    - mass_mat @ solutions[:, i - 1]
                )
                solutions[:, i + 1] = spsolve(A, b)
        elif cst_model == "cst_quad9_rot4":
            eqs_u, eqs_w, eqs_s = get_variables_eqs(assem_op)
            m_uu, k_uu, k_ww, k_us, k_ws, f_u, f_w = decouple_global_matrices(
                mass_mat, stiff_mat, rhs, eqs_u, eqs_w, eqs_s
            )

            inv_k_ww = inv(k_ww)

            A = k_ws.T @ inv_k_ww @ k_ws
            inv_A = inv(A)
            B = k_us @ inv_A @ k_us.T
            C = k_us @ inv_A @ k_ws.T @ inv_k_ww
            for i in tqdm(range(1, n_t_iters - 1), desc="iterations"):
                u_i_1 = solutions[eqs_u, i - 1]
                u_i = solutions[eqs_u, i]

                M = m_uu + dt**2 * (k_uu + B)
                b = dt**2 * (f_u + C @ f_w) + m_uu @ (2 * u_i - u_i_1)

                solutions[eqs_u, i + 1] = spsolve(M, b)

                solutions[eqs_s, i + 1] = inv_A @ (
                    k_us.T @ solutions[eqs_u, i + 1] - k_ws.T @ inv_k_ww @ f_w
                )
                solutions[eqs_w, i + 1] = inv_k_ww @ (
                    f_w + k_ws @ solutions[eqs_s, i + 1]
                )

        save_solution_files(
            bc_array, solutions, files_dict, mass_mat, stiff_mat, return_matrices
        )
        if return_matrices:
            return bc_array, solutions, mass_mat, stiff_mat, nodes, elements
        return bc_array, solutions, nodes, elements


def retrieve_solution(
    geometry_type: str,
    geometry_mesh_params: dict,
    cst_model: str,
    constraints_loads: str,
    materials: np.ndarray,
    scenario_to_solve: SOLUTION_TYPES,
    force_reprocess: bool = False,
    custom_str: str = "",
    dt: float | None = None,
    n_t_iter: int | None = None,
    initial_state: np.ndarray | dict | None = None,
    cons_loads_fcn_params: dict = {},
    return_matrices: bool = False,
):
    files_dict = generate_solution_filenames(
        geometry_type,
        cst_model,
        constraints_loads,
        scenario_to_solve,
        geometry_mesh_params,
        custom_str=custom_str,
        return_matrices=return_matrices,
    )
    cons_loads_fcn = SYSTEMS[constraints_loads]

    if check_solution_files_exists(files_dict) and not force_reprocess:
        _, elements, nodes, _ = _load_mesh(
            files_dict["mesh"], cons_loads_fcn, cons_loads_fcn_params
        )
        solution_structures = load_solutions(
            files_dict, scenario_to_solve, return_matrices
        )
        complete_response = (*solution_structures, nodes, elements)

    else:
        complete_response = _compute_solution(
            geometry_type,
            geometry_mesh_params,
            cons_loads_fcn_params,
            files_dict,
            cst_model,
            cons_loads_fcn,
            materials,
            scenario_to_solve,
            dt,
            n_t_iter,
            initial_state,
            return_matrices,
        )

    return complete_response
