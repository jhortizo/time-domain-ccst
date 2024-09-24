import os
import warnings

import numpy as np
import sympy as sp
from continuum_mechanics.solids import c_cst
from scipy.sparse.linalg import inv, spsolve
from solidspy.assemutil import assembler, loadasem
from tqdm import tqdm

from time_domain_ccst.constants import MESHES_FOLDER, SOLUTIONS_FOLDER
from time_domain_ccst.constraints_loads_creators import borders_fixed
from time_domain_ccst.cst_utils import (
    assem_op_cst_quad9_rot4,
    cst_quad9_rot4,
    get_variables_eqs,
)
from time_domain_ccst.fem_solver import _load_mesh
from time_domain_ccst.gmesher import _create_square_mesh

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


x, y, t = sp.symbols("x y t")
E = 1
nu = 0.3
rho = 1
eta = 1e-5
omega = 1


def calculate_body_force_fcn_continuum_mechanics(u: sp.Matrix) -> sp.Matrix:
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    ccst_operator = c_cst(u, (lambda_, mu, eta)) - rho * u.diff(
        t, 2
    )  # TODO check why is taking too long

    f = -ccst_operator.row_del(2)
    f_lambdified = sp.lambdify((x, y, t), f, "numpy")

    return f_lambdified, f


def generate_load_mesh(mesh_size: float, mesh_file: str):
    _create_square_mesh(1.0, mesh_size, mesh_file)
    cons, elements, nodes, loads = _load_mesh(mesh_file, borders_fixed, {})
    return cons, elements, nodes, loads


def impose_body_force_loads(
    loads: np.ndarray, nodes: np.ndarray, body_force_fcn: callable, current_time: float
):
    loads[:, 0] = np.arange(len(nodes))  # specify nodes

    # correctly reorder array from (2, 1, nnodes) to (nnodes, 2)
    body_force_nodes = body_force_fcn(nodes[:, 1], nodes[:, 2], current_time)
    body_force_nodes = np.squeeze(body_force_nodes)
    body_force_nodes = np.swapaxes(body_force_nodes, 0, 1)

    # there is no problem in applying loads on the boudaries because
    # they are already marked as fixed, so those loads will be ignored
    loads[:, 1:3] = body_force_nodes

    return loads


def solve_manufactured_solution(
    mesh_size: float,
    dt: float,
    n_t_iters: int,
    body_force_fcn: callable,
    u_fnc: callable,
    force_reprocess: bool = False,
    custom_string: str = "",
):
    solution_identifier = (
        f"manufactured_solution_dynamic_mesh_size_{mesh_size}{custom_string}"
    )
    mesh_file = f"{MESHES_FOLDER}/{solution_identifier}.msh"
    solution_file = f"{SOLUTIONS_FOLDER}/{solution_identifier}-solution.csv"
    bc_array_file = f"{SOLUTIONS_FOLDER}/{solution_identifier}-bc_array.csv"
    rhs_file = f"{SOLUTIONS_FOLDER}/{solution_identifier}-rhs.csv"

    files_list = [mesh_file, solution_file, bc_array_file]

    # TODO: this is being redone over and over again, consider refactoring this as well
    cons, elements, nodes, loads = generate_load_mesh(mesh_size, mesh_file)
    print("Mesh size:", len(elements), " elements")


    if (
        not all([os.path.exists(this_file) for this_file in files_list])
        or force_reprocess
    ):
        mats = [
            E,
            nu,
            eta,
            rho,
        ]

        mats = np.array([mats])

        assem_op, bc_array, neq = assem_op_cst_quad9_rot4(cons, elements)
        stiff_mat, mass_mat = assembler(
            elements, mats, nodes, neq, assem_op, uel=cst_quad9_rot4
        )

        u_initial = u_fnc(nodes[:, 1], nodes[:, 2], 0)
        u_initial = np.squeeze(u_initial)
        u_initial = np.swapaxes(u_initial, 0, 1)

        initial_state = inverse_complete_disp(
            bc_array, nodes, u_initial, len(elements), ndof_node=2
        )

        # check the initial state is correct

        solutions = np.zeros((neq, n_t_iters))
        solutions[:, 0] = initial_state  # assume constant behavior in first steps
        solutions[:, 1] = initial_state

        eqs_u, eqs_w, eqs_s = get_variables_eqs(assem_op)
        m_uu, k_uu, k_ww, k_us, k_ws = decouple_global_matrices_only(
            mass_mat, stiff_mat, eqs_u, eqs_w, eqs_s
        )

        inv_k_ww = inv(k_ww)

        A = k_ws.T @ inv_k_ww @ k_ws
        inv_A = inv(A)
        B = k_us @ inv_A @ k_us.T
        C = k_us @ inv_A @ k_ws.T @ inv_k_ww
        for i in tqdm(range(1, n_t_iters - 1), desc="iterations"):
            # get loads for current time
            current_time = (i + 1) * dt

            # TODO: I could do this outside vectorized, to optimize
            loads = impose_body_force_loads(loads, nodes, body_force_fcn, current_time)
            rhs = loadasem(loads, bc_array, neq)
            # approximation to eval the function weighted by the form functions
            rhs = mass_mat @ rhs
            f_u = rhs[eqs_u]
            # f_u = np.zeros_like(f_u) # to debug
            f_w = rhs[eqs_w]

            u_i_1 = solutions[eqs_u, i - 1]
            u_i = solutions[eqs_u, i]

            M = m_uu + dt**2 * (k_uu + B)
            b = dt**2 * (f_u + C @ f_w) + m_uu @ (2 * u_i - u_i_1)
            # M = m_uu + dt**2 * (k_uu)
            # b = dt**2 * (f_u) + m_uu @ (2 * u_i - u_i_1)

            solutions[eqs_u, i + 1] = spsolve(M, b)

            # from solidspy.postprocesor import complete_disp, plot_node_field
            # # loads_field = complete_disp(bc_array, nodes, rhs, ndof_node=2)
            # # plot_node_field(loads_field, nodes, elements, title=["loads_x", "loads_y "])
            # u_field = complete_disp(bc_array, nodes, solutions[:, i], ndof_node=2)
            # plot_node_field(u_field, nodes, elements, title=["u_x", "u_y "])

        np.savetxt(solution_file, solutions, delimiter=",")
        np.savetxt(bc_array_file, bc_array, delimiter=",")
        np.savetxt(rhs_file, rhs, delimiter=",")

    else:
        bc_array = np.loadtxt(bc_array_file, delimiter=",").astype(np.int64)
        bc_array = bc_array.reshape(-1, 1) if bc_array.ndim == 1 else bc_array
        solutions = np.loadtxt(solution_file, delimiter=",")
        rhs = np.loadtxt(rhs_file, delimiter=",")
        cons, elements, nodes, loads = generate_load_mesh(mesh_size, mesh_file)

    return bc_array, solutions, nodes, elements, rhs


def inverse_complete_disp(bc_array, nodes, sol_complete, len_elements, ndof_node=2):
    nnodes = nodes.shape[0]
    sol = np.zeros(bc_array.max() + len_elements + 1, dtype=float)
    for row in range(nnodes):
        for col in range(ndof_node):
            cons = bc_array[row, col]
            if cons != -1:
                sol[cons] = sol_complete[row, col]
    return sol


def decouple_global_matrices_only(mass_mat, stiff_mat, eqs_u, eqs_w, eqs_s):
    """Decouple the global matrices"""
    pass
    m_uu = mass_mat[np.ix_(eqs_u, eqs_u)]
    k_uu = stiff_mat[np.ix_(eqs_u, eqs_u)]
    k_ww = stiff_mat[np.ix_(eqs_w, eqs_w)]
    k_us = stiff_mat[np.ix_(eqs_u, eqs_s)]
    k_ws = -1 * stiff_mat[np.ix_(eqs_w, eqs_s)]

    return m_uu, k_uu, k_ww, k_us, k_ws
