"""
In this module we define the functions that create constraints and loads for
different cases tested the library.
"""

import numpy as np


def lower_roller_left_roller_upper_force(line3, cell_data, npts):
    """
    The lower and left borders are fixed in y and x respectively. The upper
    border has a force in the y direction.
    """

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

    return cons, loads


def borders_fixed(line3, cell_data, npts):
    lower_border = set(line3[cell_data["line3"]["gmsh:physical"] == 1].flatten())
    right_border = set(line3[cell_data["line3"]["gmsh:physical"] == 2].flatten())
    upper_border = set(line3[cell_data["line3"]["gmsh:physical"] == 3].flatten())
    left_border = set(line3[cell_data["line3"]["gmsh:physical"] == 4].flatten())

    cons = np.zeros((npts, 3), dtype=int)
    cons[list(lower_border), :] = -1
    cons[list(right_border), :] = -1
    cons[list(upper_border), :] = -1
    cons[list(left_border), :] = -1

    loads = np.zeros((npts, 4))  # empty loads

    return cons, loads


SYSTEMS = {
    "lower_roller_left_roller_upper_force": lower_roller_left_roller_upper_force,
    'borders_fixed': borders_fixed
}
