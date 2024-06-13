import os

import numpy as np

from .constants import MESHES_FOLDER, SOLUTIONS_FOLDER


def _parse_solution_identifier(geometry_type, cst_model, constraints_loads, params, custom_str):
    "Returnss strign associated with run parameters"

    params_str = [
        str(this_key) + "_" + str(this_value) for this_key, this_value in params.items()
    ]

    filename = (
        geometry_type
        + "-"
        + cst_model
        + "-"
        + constraints_loads
        + "-"
        + "-".join(params_str)
        + custom_str
    )

    return filename


def generate_solution_filenames(
    geometry_type: str,
    cst_model: str,
    constraints_loads: str,
    eigensolution: bool,
    params: dict,
    custom_str,
):
    "Returns filenames for solution files"
    solution_id = _parse_solution_identifier(
        geometry_type, cst_model, constraints_loads, params, custom_str
    )
    bc_array_file = f"{SOLUTIONS_FOLDER}/{solution_id}-bc_array.csv"
    mesh_file = f"{MESHES_FOLDER}/{solution_id}.msh"
    if eigensolution:
        eigvals_file = f"{SOLUTIONS_FOLDER}/{solution_id}-eigvals.csv"
        eigvecs_file = f"{SOLUTIONS_FOLDER}/{solution_id}-eigvecs.csv"
        return {
            "bc_array": bc_array_file,
            "eigvals": eigvals_file,
            "eigvecs": eigvecs_file,
            "mesh": mesh_file,
        }

    else:
        solution_file = f"{SOLUTIONS_FOLDER}/{solution_id}-solution.csv"
        return {
            "bc_array": bc_array_file,
            "solution": solution_file,
            "mesh": mesh_file,
        }


def check_solution_files_exists(files_dict):
    "Checks if solution files exist"
    return all([os.path.exists(this_file) for this_file in files_dict.values()])


def load_solution_files(files_dict):
    "Loads solution files"
    bc_array = np.loadtxt(files_dict["bc_array"], delimiter=",", dtype=int)
    bc_array = bc_array.reshape(-1, 1) if bc_array.ndim == 1 else bc_array
    solution = np.loadtxt(files_dict["solution"], delimiter=",")
    return bc_array, solution


def load_eigensolution_files(files_dict):
    "Loads solution files"
    bc_array = np.loadtxt(files_dict["bc_array"], delimiter=",", dtype=int)
    bc_array = bc_array.reshape(-1, 1) if bc_array.ndim == 1 else bc_array
    eigvals = np.loadtxt(files_dict["eigvals"], delimiter=",")
    eigvecs = np.loadtxt(files_dict["eigvecs"], delimiter=",")
    return bc_array, eigvals, eigvecs


def save_solution_files(bc_array, solution, files_dict):
    "Saves solution files"
    np.savetxt(files_dict["bc_array"], bc_array, delimiter=",")
    np.savetxt(files_dict["solution"], solution, delimiter=",")


def save_eigensolution_files(bc_array, eigvals, eigvecs, files_dict):
    "Saves solution files for eigenvalues and eigenvectors problem"
    np.savetxt(files_dict["bc_array"], bc_array, delimiter=",")
    np.savetxt(files_dict["eigvals"], eigvals, delimiter=",")
    np.savetxt(files_dict["eigvecs"], eigvecs, delimiter=",")


def postprocess_eigsolution(eigvals, eigvecs):
    "Postprocesses eigenvalues and eigenvectors"
    # check eigvals are real
    if not np.allclose(eigvals.imag, 0):
        print("Eigenvalues are not real")
    
    order = np.argsort(eigvals)
    eigvals = np.sort(eigvals).real
    eigvecs = eigvecs[:, order].real
    return eigvals, eigvecs
