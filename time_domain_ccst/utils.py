import os

import numpy as np

from .constants import MESHES_FOLDER, SOLUTION_TYPES, SOLUTIONS_FOLDER


def _parse_solution_identifier(
    geometry_type, cst_model, constraints_loads, params, custom_str
):
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
    scenario_to_solve: SOLUTION_TYPES,
    params: dict,
    custom_str,
):
    "Returns filenames for solution files"
    solution_id = _parse_solution_identifier(
        geometry_type, cst_model, constraints_loads, params, custom_str
    )
    bc_array_file = f"{SOLUTIONS_FOLDER}/{solution_id}-bc_array.csv"
    mesh_file = f"{MESHES_FOLDER}/{solution_id}.msh"

    solution_files = {
        "bc_array": bc_array_file,
        "mesh": mesh_file,
    }

    if scenario_to_solve == "static":
        solution_files["solution"] = f"{SOLUTIONS_FOLDER}/{solution_id}-solution.csv"
    elif scenario_to_solve == "eigenproblem":
        solution_files["eigvals"] = f"{SOLUTIONS_FOLDER}/{solution_id}-eigvals.csv"
        solution_files["eigvecs"] = f"{SOLUTIONS_FOLDER}/{solution_id}-eigvecs.csv"
    elif scenario_to_solve == "time-marching":
        solution_files["solution"] = (
            f"{SOLUTIONS_FOLDER}/{solution_id}-time-solutions.csv"
        )

    return solution_files


def check_solution_files_exists(files_dict):
    "Checks if solution files exist"
    return all([os.path.exists(this_file) for this_file in files_dict.values()])


def load_solutions(files_dict, scenario_to_solve):
    "Loads solution files"
    load_solutions = {
        "static": load_static_solution_files,
        "eigenproblem": load_eigensolution_files,
        "time-marching": load_dynamic_solution_files,
    }
    file_loading_fcn = load_solutions[scenario_to_solve]
    solution_structures = file_loading_fcn(files_dict)
    return solution_structures


def load_static_solution_files(files_dict):
    "Loads solution files"
    bc_array = np.loadtxt(files_dict["bc_array"], delimiter=",", dtype=int)
    bc_array = bc_array.reshape(-1, 1) if bc_array.ndim == 1 else bc_array
    solution = np.loadtxt(files_dict["solution"], delimiter=",")
    return bc_array, solution


def load_dynamic_solution_files(files_dict):
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
    save_solution(bc_array, files_dict["bc_array"])
    save_solution(solution, files_dict["solution"])


def save_eigensolution_files(bc_array, eigvals, eigvecs, files_dict):
    "Saves solution files for eigenvalues and eigenvectors problem"
    save_solution(bc_array, files_dict["bc_array"])
    save_solution(eigvals, files_dict["eigvals"])
    save_solution(eigvecs, files_dict["eigvecs"])


def save_solution(data, filename):
    "Saves a solution file"
    np.savetxt(filename, data, delimiter=",")


def postprocess_eigsolution(eigvals, eigvecs):
    "Postprocesses eigenvalues and eigenvectors"
    # check eigvals are real
    if not np.allclose(eigvals.imag, 0):
        print("Eigenvalues are not real")

    order = np.argsort(eigvals)
    eigvals = np.sort(eigvals).real
    eigvecs = eigvecs[:, order].real
    return eigvals, eigvecs
