"""
Solve for wave propagation in classical mechanics in the given domain.
"""

import meshio
import numpy as np
from scipy.sparse.linalg import spsolve
from solidspy.assemutil import assembler, loadasem
from solidspy_uels.solidspy_uels import assem_op_cst, cst_quad9

from .constants import MATERIAL_PARAMETERS
from .cst_utils import assem_op_cst_quad9_rot4, cst_quad9_rot4
from .gmesher import create_mesh
from .utils import (
    check_solution_files_exists,
    generate_solution_filenames,
    load_solution_files,
    save_solution_files,
)

cst_model_functions = {
    "cst_quad9_rot4": (assem_op_cst_quad9_rot4, cst_quad9_rot4),
    "cst_quad9": (assem_op_cst, cst_quad9),
}


def _load_mesh(mesh_file: str) -> tuple[np.array, np.array, np.array, np.array]:
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

    # Constraints and Loads TODO: decouple this from solver
    line3 = cells["line3"]
    cell_data = mesh.cell_data
    cons = np.zeros((npts, 3), dtype=int)
    # lower border is fixed in y and roll in x
    # left border is fixed in x and roll in y
    lower_border = set(line3[cell_data["line3"]["gmsh:physical"] == 1].flatten())
    left_border = set(line3[cell_data["line3"]["gmsh:physical"] == 4].flatten())

    upper_border = set(line3[cell_data["line3"]["gmsh:physical"] == 3].flatten())

    cons[list(lower_border), 1] = -1
    cons[list(left_border), 0] = -1
    loads = np.zeros((npts, 4))  # empty loads
    loads[:, 0] = np.arange(npts)  # specify nodes

    loads[list(upper_border), 1 + 1] = 100  # force in y direction

    return cons, elements, nodes, loads


def _compute_solution(
    geometry_type: str, params: dict, files_dict: dict, cst_model="cst_quad9_rot4"
):
    assem_op, cst_element = cst_model_functions[cst_model]
    omega = 1

    mats = [
        MATERIAL_PARAMETERS["E"],
        MATERIAL_PARAMETERS["NU"],
        MATERIAL_PARAMETERS["ETA"],
        MATERIAL_PARAMETERS["RHO"],
    ]  # order imposed by cst_quad9

    mats = np.array([mats])

    create_mesh(geometry_type, params, files_dict["mesh"])

    cons, elements, nodes, loads = _load_mesh(files_dict["mesh"])
    # Assembly
    assem_op, bc_array, neq = assem_op(cons, elements)
    stiff_mat, mass_mat = assembler(
        elements, mats, nodes, neq, assem_op, uel=cst_element
    )

    rhs = loadasem(loads, bc_array, neq)
    # Solution
    solution = spsolve(stiff_mat - omega**2 * mass_mat, rhs)

    save_solution_files(bc_array, solution, files_dict)

    return bc_array, solution, nodes, elements


def retrieve_solution(
    geometry_type: str, params: dict, cst_model: str, force_reprocess: bool = False
):
    # TODO: refactor to use third party cache instead of file system
    files_dict = generate_solution_filenames(geometry_type, params)

    if check_solution_files_exists(files_dict) and not force_reprocess:
        bc_array, solution = load_solution_files(files_dict)
        _, elements, nodes, _ = _load_mesh(files_dict["mesh"])

    else:
        bc_array, solution, nodes, elements = _compute_solution(
            geometry_type, params, files_dict, cst_model
        )

    return bc_array, solution, nodes, elements
