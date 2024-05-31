import os

import numpy as np

from .constants import MESHES_FOLDER, SOLUTIONS_FOLDER


def _parse_solution_identifier(geometry_type, cst_model, constraints_loads, params):
    "Returnss strign associated with run parameters"

    params_str = [
        str(this_key) + "_" + str(this_value)
        for this_key, this_value in params.items()
    ] 

    filename = geometry_type + cst_model + constraints_loads + "-" + "-".join(params_str)

    return filename


def generate_solution_filenames(geometry_type, cst_model, constraints_loads, params):
    "Returns filenames for solution files"
    solution_id = _parse_solution_identifier(geometry_type, cst_model, constraints_loads, params)
    bc_array_file = f"{SOLUTIONS_FOLDER}/{solution_id}-bc_array.csv"
    solution_file = f"{SOLUTIONS_FOLDER}/{solution_id}-solution.csv"
    mesh_file = f"{MESHES_FOLDER}/{solution_id}.msh"
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


def save_solution_files(bc_array, solution, files_dict):
    "Saves solution files"
    np.savetxt(files_dict["bc_array"], bc_array, delimiter=",")
    np.savetxt(files_dict["solution"], solution, delimiter=",")
