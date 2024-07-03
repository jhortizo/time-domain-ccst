import os
from typing import Literal

MESHES_FOLDER = "data/meshes"
SOLUTIONS_FOLDER = "data/solutions"
IMAGES_FOLDER = "data/images"

QUAD9_ROT4_ELEMENT_DOFS_IDS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    16,
    18,
    19,
    21,
    22,
    24,
    25,
]  # toss the 14th, 17th, 20th, 23rd, 26th dofs, as are the rotation ones for the non-vertex nodes

# Create the folders if they don't exist
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
abs_meshes_folder = os.path.join(current_dir, MESHES_FOLDER)
abs_solutions_folder = os.path.join(current_dir, SOLUTIONS_FOLDER)
abs_images_folder = os.path.join(current_dir, IMAGES_FOLDER)
os.makedirs(abs_meshes_folder, exist_ok=True)
os.makedirs(abs_solutions_folder, exist_ok=True)
os.makedirs(abs_images_folder, exist_ok=True)

# some custom types
SOLUTION_TYPES = Literal["static", "eigenproblem", "time-marching"]