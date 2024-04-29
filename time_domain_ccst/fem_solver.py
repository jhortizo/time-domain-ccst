"""
Solve for wave propagation in classical mechanics in the given domain.
"""

import meshio
import numpy as np
import solidspy.assemutil as ass
from scipy.sparse.linalg import eigsh
from solidspy.assemutil import assembler
from solidspy_uels.solidspy_uels import assem_op_cst, cst_quad9

from .constants import MATERIAL_PARAMETERS
from .gmesher import create_mesh
from .utils import (
    check_solution_files_exists,
    generate_solution_filenames,
    load_solution_files,
    save_solution_files,
)


def _load_mesh(mesh_file):
    mesh = meshio.read(mesh_file)

    points = mesh.points
    cells = mesh.cells
    quad9 = cells["quad9"]
    line3 = cells["line3"]
    npts = points.shape[0]
    nels = quad9.shape[0]

    nodes = np.zeros((npts, 3))
    nodes[:, 1:] = points[:, 0:2]

    # Constraints
    line_nodes = list(set(line3.flatten()))
    cons = np.zeros((npts, 3), dtype=int)
    cons[line_nodes, :] = -1

    # Elements
    elements = np.zeros((nels, 3 + 9), dtype=int)
    elements[:, 0] = range(nels)
    elements[:, 1] = 4
    elements[
        :, 3:
    ] = quad9  # the first 3 cols correspond to material params and elements params
    # the remaining are the nodes ids

    return cons, elements, nodes


def _compute_solution(geometry_type: str, params: dict, files_dict: dict):
    mats = [
        MATERIAL_PARAMETERS["E"],
        MATERIAL_PARAMETERS["NU"],
        MATERIAL_PARAMETERS["ETA"],
        MATERIAL_PARAMETERS["RHO"],
    ]  # order imposed by elast_tri6

    mats = np.array([mats])

    create_mesh(geometry_type, params, files_dict["mesh"])

    cons, elements, nodes = _load_mesh(files_dict["mesh"])
    # Assembly
    assem_op, bc_array, neq = assem_op_cst(cons, elements)
    stiff_mat, mass_mat = assembler(elements, mats, nodes, neq, assem_op, uel=cst_quad9)

    # Solution
    eigvals, eigvecs = eigsh(
        stiff_mat, M=mass_mat, k=stiff_mat.shape[0] - 1, which="SM"
    )

    save_solution_files(bc_array, eigvals, eigvecs, files_dict)

    return bc_array, eigvals, eigvecs, nodes, elements


def retrieve_solution(geometry_type: str, params: dict, force_reprocess: bool = False):
    files_dict = generate_solution_filenames(geometry_type, params)

    if check_solution_files_exists(files_dict) and not force_reprocess:
        bc_array, eigvals, eigvecs = load_solution_files(files_dict)
        _, elements, nodes = _load_mesh(files_dict["mesh"])

    else:
        bc_array, eigvals, eigvecs, nodes, elements = _compute_solution(
            geometry_type, params, files_dict
        )

    return bc_array, eigvals, eigvecs, nodes, elements
