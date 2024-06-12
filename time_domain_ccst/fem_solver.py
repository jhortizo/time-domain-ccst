"""
Solve for wave propagation in classical mechanics in the given domain.
"""

import meshio
import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import spsolve
from solidspy.assemutil import assembler, loadasem
from solidspy_uels.solidspy_uels import assem_op_cst, cst_quad9

from .constraints_loads_creators import SYSTEMS
from .cst_utils import assem_op_cst_quad9_rot4, cst_quad9_rot4
from .gmesher import create_mesh
from .utils import (
    check_solution_files_exists,
    generate_solution_filenames,
    load_eigensolution_files,
    load_solution_files,
    postprocess_eigsolution,
    save_eigensolution_files,
    save_solution_files,
)

cst_model_functions = {
    "cst_quad9_rot4": (assem_op_cst_quad9_rot4, cst_quad9_rot4),
    "cst_quad9": (assem_op_cst, cst_quad9),
}


def _load_mesh(
    mesh_file: str, cons_loads_fcn: callable
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
    cons, loads = cons_loads_fcn(line3, cell_data, npts)

    return cons, elements, nodes, loads


def _compute_solution(
    geometry_type: str,
    params: dict,
    files_dict: dict,
    cst_model: str,
    cons_loads_fcn: callable,
    materials: np.ndarray,
    eigensolution: bool,
):
    assem_op, cst_element = cst_model_functions[cst_model]

    create_mesh(geometry_type, params, files_dict["mesh"])

    cons, elements, nodes, loads = _load_mesh(files_dict["mesh"], cons_loads_fcn)
    # Assembly

    can_be_sparse = not (eigensolution)
    assem_op, bc_array, neq = assem_op(cons, elements)
    stiff_mat, mass_mat = assembler(
        elements, materials, nodes, neq, assem_op, uel=cst_element, sparse=can_be_sparse
    )

    if eigensolution:
        eigvals, eigvecs = eig(stiff_mat, b=mass_mat)
        eigvals, eigvecs = postprocess_eigsolution(eigvals, eigvecs)
        save_eigensolution_files(bc_array, eigvals, eigvecs, files_dict)
        return bc_array, eigvals, eigvecs, nodes, elements

    else:
        # omega = 1
        rhs = loadasem(loads, bc_array, neq)
        solution = spsolve(stiff_mat, rhs)
        save_solution_files(bc_array, solution, files_dict)
        return bc_array, solution, nodes, elements


def retrieve_solution(
    geometry_type: str,
    params: dict,
    cst_model: str,
    constraints_loads: str,
    materials: np.ndarray,
    eigensolution: bool = False,
    force_reprocess: bool = False,
):
    files_dict = generate_solution_filenames(
        geometry_type, cst_model, constraints_loads, eigensolution, params
    )
    cons_loads_fcn = SYSTEMS[constraints_loads]

    if check_solution_files_exists(files_dict) and not force_reprocess:
        _, elements, nodes, _ = _load_mesh(files_dict["mesh"], cons_loads_fcn)

        if eigensolution:
            bc_array, eigvals, eigvecs = load_eigensolution_files(files_dict)
        else:
            bc_array, solution = load_solution_files(files_dict)

    else:
        if eigensolution:
            bc_array, eigvals, eigvecs, nodes, elements = _compute_solution(
                geometry_type,
                params,
                files_dict,
                cst_model,
                cons_loads_fcn,
                materials,
                eigensolution,
            )
        else:
            bc_array, solution, nodes, elements = _compute_solution(
                geometry_type,
                params,
                files_dict,
                cst_model,
                cons_loads_fcn,
                materials,
                eigensolution,
            )

    if eigensolution:
        return bc_array, eigvals, eigvecs, nodes, elements
    else:
        return bc_array, solution, nodes, elements
