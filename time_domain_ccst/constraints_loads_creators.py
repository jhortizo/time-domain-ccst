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


def quarter_ring_rollers_axial_load(line3, cell_data, npts):
    """
    TODO: properly describe this, maybe add some figures to illustrate...
    """

    up_border = set(line3[cell_data["line3"]["gmsh:physical"] == 3].flatten())
    low_border = set(line3[cell_data["line3"]["gmsh:physical"] == 1].flatten())
    inner_cicle = set(line3[cell_data["line3"]["gmsh:physical"] == 2].flatten())

    cons = np.zeros((npts, 3), dtype=int)
    cons[list(up_border), 0] = -1
    cons[list(low_border), 1] = -1

    loads = np.zeros((npts, 4))  # empty loads
    loads[:, 0] = np.arange(npts)  # specify nodes

    axial_node_load = 100
    loads[list(inner_cicle), 0 + 1] = axial_node_load / 2  # force in x direction
    loads[list(inner_cicle), 1 + 1] = axial_node_load / 2  # force in y direction

    return cons, loads


def cantilever_support_load(line3, cell_data, npts, load=-1):
    """
    TODO: properly describe this, maybe add some figures to illustrate...
    """

    left_border = list(set(line3[cell_data["line3"]["gmsh:physical"] == 4].flatten()))
    right_border = list(set(line3[cell_data["line3"]["gmsh:physical"] == 2].flatten()))

    cons = np.zeros((npts, 3), dtype=int)
    cons[left_border, :] = -1

    loads = np.zeros((npts, 4))  # empty loads
    loads[:, 0] = np.arange(npts)  # specify nodes

    loads[list(right_border), 1 + 1] = load  # force in y direction

    return cons, loads


def cantilever_support(line3, cell_data, npts):
    """
    TODO: properly describe this, maybe add some figures to illustrate...
    """

    left_border = set(line3[cell_data["line3"]["gmsh:physical"] == 4].flatten())

    cons = np.zeros((npts, 3), dtype=int)
    cons[list(left_border), :] = -1

    loads = np.zeros((npts, 4))  # empty loads
    loads[:, 0] = np.arange(npts)  # specify nodes

    return cons, loads


def cantilever_support_classical(line3, cell_data, npts):
    """
    TODO: properly describe this, maybe add some figures to illustrate...

    Classical refers to only add 2 dofs per node, instead of 3.
    """

    left_border = set(line3[cell_data["line3"]["gmsh:physical"] == 4].flatten())

    cons = np.zeros((npts, 2), dtype=int)
    cons[list(left_border), :] = -1

    loads = np.zeros((npts, 3))  # empty loads
    loads[:, 0] = np.arange(npts)  # specify nodes

    return cons, loads


def plate_hole_rollers_load(line3, cell_data, npts):
    """
    TODO: properly describe this, maybe add some figures to illustrate...
    """

    lower_border = set(line3[cell_data["line3"]["gmsh:physical"] == 1].flatten())
    left_border = set(line3[cell_data["line3"]["gmsh:physical"] == 4].flatten())
    right_border = set(line3[cell_data["line3"]["gmsh:physical"] == 2].flatten())

    cons = np.zeros((npts, 3), dtype=int)
    cons[list(left_border), 0] = -1
    cons[list(lower_border), 1] = -1

    loads = np.zeros((npts, 4))  # empty loads
    loads[:, 0] = np.arange(npts)  # specify nodes

    loads[list(right_border), 0 + 1] = 100  # force in x direction

    return cons, loads


def circle_borders_fixed(line3, cell_data, npts):
    """
    TODO: properly describe this, maybe add some figures to illustrate...
    """

    upper_circle = set(line3[cell_data["line3"]["gmsh:physical"] == 1].flatten())
    lower_circle = set(line3[cell_data["line3"]["gmsh:physical"] == 2].flatten())

    cons = np.zeros((npts, 3), dtype=int)
    cons[list(upper_circle), 0:2] = -1
    cons[list(lower_circle), 0:2] = -1

    loads = np.zeros((npts, 4))  # empty loads

    return cons, loads


def no_constraints_no_loads(line3, cell_data, npts):
    cons = np.zeros((npts, 3), dtype=int)
    loads = np.zeros((npts, 4))

    return cons, loads


SYSTEMS = {
    "lower_roller_left_roller_upper_force": lower_roller_left_roller_upper_force,
    "borders_fixed": borders_fixed,
    "quarter_ring_rollers_axial_load": quarter_ring_rollers_axial_load,
    "cantilever_support_load": cantilever_support_load,
    "cantilever_support": cantilever_support,
    "cantilever_support_classical": cantilever_support_classical,
    "plate_hole_rollers_load": plate_hole_rollers_load,
    "circle_borders_fixed": circle_borders_fixed,
    "floating": no_constraints_no_loads,
}
