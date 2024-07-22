import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from continuum_mechanics.solids import c_cst
from scipy.sparse.linalg import spsolve
from solidspy.assemutil import assembler, loadasem

from time_domain_ccst.constants import MESHES_FOLDER, SOLUTIONS_FOLDER
from time_domain_ccst.constraints_loads_creators import borders_fixed
from time_domain_ccst.cst_utils import assem_op_cst_quad9_rot4, cst_quad9_rot4
from time_domain_ccst.fem_solver import _load_mesh
from time_domain_ccst.gmesher import _create_square_mesh

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


x, y = sp.symbols("x y")
E = 1
nu = 0.3
rho = 1
eta = 1
omega = 1


def manufactured_solution() -> sp.Matrix:
    u = sp.Matrix(
        [
            x * (1 - x) * y * (1 - y) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y),
            x * (1 - x) * y * (1 - y) * sp.cos(sp.pi * x) * sp.cos(sp.pi * y),
            0,
        ]
    )
    # TODO: for some reason here the u.row_del(2) decided not to work, try and fix it
    u_2d = sp.Matrix(
        [
            x * (1 - x) * y * (1 - y) * sp.sin(sp.pi * x) * sp.sin(sp.pi * y),
            x * (1 - x) * y * (1 - y) * sp.cos(sp.pi * x) * sp.cos(sp.pi * y),
        ]
    )

    # u = sp.Matrix(
    #     [sp.sin(sp.pi * x) * sp.sin(sp.pi * y), sp.sin(sp.pi * y) * sp.sin(sp.pi * x)]
    # )

    u_lambdified = sp.lambdify((x, y), u_2d, "numpy")
    return u, u_lambdified


def calculate_body_force_fcn_manually(u: sp.Matrix) -> sp.Matrix:
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    c1_squared = (lambda_ + 2 * mu) / rho
    c2_squared = mu / rho
    l_squared = eta / mu
    div = sp.diff(u[0], x) + sp.diff(u[1], y)

    curl_u = sp.diff(u[1], x) - sp.diff(u[0], y)
    double_curl_u = sp.Matrix([sp.diff(curl_u, y), -sp.diff(curl_u, x)])

    laplacian_u = sp.Matrix(
        [
            sp.diff(double_curl_u[0], x, 2) + sp.diff(double_curl_u[0], y, 2),
            sp.diff(double_curl_u[1], x, 2) + sp.diff(double_curl_u[1], y, 2),
        ]
    )

    term1 = c1_squared * sp.Matrix([sp.diff(div, x), sp.diff(div, y)])  # checked
    term2 = c2_squared * (double_curl_u - l_squared * laplacian_u)

    # rhs = -(omega**2) * u
    # rhs = 0

    f = -(term1 - term2)
    # f_simplified = f.applyfunc(sp.simplify)

    f_lambdified = sp.lambdify((x, y), f, "numpy")

    return f_lambdified, f


def calculate_body_force_fcn_continuum_mechanics(u: sp.Matrix) -> sp.Matrix:
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    ccst_operator = c_cst(u, (lambda_, mu, eta))

    f = -ccst_operator.row_del(2)
    f_lambdified = sp.lambdify((x, y), f, "numpy")

    return f_lambdified, f


def check_manual_vs_continuum_mechanics(
    f_m: sp.Matrix, f_cm: sp.Matrix, body_force_fcn_manual, body_force_fcn_cm
) -> None:
    # symbolic comparison
    difference = sp.simplify(f_m - f_cm)  # the expressions are in the e-15 order
    print(difference)

    # numerical comparison
    nx, ny = (3, 2)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)

    val_cm = body_force_fcn_cm(xv, yv)
    val_m = body_force_fcn_manual(xv, yv)
    comp_difference = val_m - val_cm

    plt.contourf(xv, yv, comp_difference[0, 0, :, :])
    plt.colorbar()

    max_error = np.max(np.abs(comp_difference))
    print("Max error:", max_error)


def generate_load_mesh(mesh_size: float, mesh_file: str):
    _create_square_mesh(1.0, mesh_size, mesh_file)
    cons, elements, nodes, loads = _load_mesh(mesh_file, borders_fixed)
    return cons, elements, nodes, loads


def impose_body_force_loads(
    loads: np.ndarray, nodes: np.ndarray, body_force_fcn: callable, elements
):
    loads[:, 0] = np.arange(len(nodes))  # specify nodes

    # correctly reorder array from (2, 1, nnodes) to (nnodes, 2)
    body_force_nodes = body_force_fcn(nodes[:, 1], nodes[:, 2])
    body_force_nodes = np.squeeze(body_force_nodes)
    body_force_nodes = np.swapaxes(body_force_nodes, 0, 1)

    # there is no problem in applying loads on the boudaries because
    # they are already marked as fixed, so those loads will be ignored
    loads[:, 1:3] = body_force_nodes

    return loads


def solve_manufactured_solution(
    mesh_size: float, body_force_fcn: callable, force_reprocess: bool = False
):
    solution_identifier = f"manufactured_solution_mesh_size_{mesh_size}"
    mesh_file = f"{MESHES_FOLDER}/{solution_identifier}.msh"
    solution_file = f"{SOLUTIONS_FOLDER}/{solution_identifier}-solution.csv"
    bc_array_file = f"{SOLUTIONS_FOLDER}/{solution_identifier}-bc_array.csv"
    rhs_file = f"{SOLUTIONS_FOLDER}/{solution_identifier}-rhs.csv"

    files_list = [mesh_file, solution_file, bc_array_file]

    # TODO: this is being redone over and over again, consider refactoring this as well
    cons, elements, nodes, loads = generate_load_mesh(mesh_size, mesh_file)

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

        loads = impose_body_force_loads(loads, nodes, body_force_fcn, elements)

        assem_op, bc_array, neq = assem_op_cst_quad9_rot4(cons, elements)
        stiff_mat, mass_mat = assembler(
            elements, mats, nodes, neq, assem_op, uel=cst_quad9_rot4
        )

        rhs = loadasem(loads, bc_array, neq)
        # approximation to eval the function weighted by the form functions
        rhs = mass_mat @ rhs
        solution = spsolve(stiff_mat, rhs)

        np.savetxt(solution_file, solution, delimiter=",")
        np.savetxt(bc_array_file, bc_array, delimiter=",")
        np.savetxt(rhs_file, rhs, delimiter=",")

    else:
        bc_array = np.loadtxt(bc_array_file, delimiter=",").astype(np.int64)
        bc_array = bc_array.reshape(-1, 1) if bc_array.ndim == 1 else bc_array
        solution = np.loadtxt(solution_file, delimiter=",")
        rhs = np.loadtxt(rhs_file, delimiter=",")
        cons, elements, nodes, loads = generate_load_mesh(mesh_size, mesh_file)

    return bc_array, solution, nodes, elements, rhs
