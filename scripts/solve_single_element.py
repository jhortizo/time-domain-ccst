"""
Hotfix (as in machete), to check the solution for a single element. The functions that are already wrapped
in the module will be modified here just to be used for this specific case.

This script is just meant to keep the analysis reproducible, but I expect it not to be a central part of the codebase.
"""

import warnings

import numpy as np
from scipy.sparse.linalg import spsolve
from solidspy.assemutil import assembler, loadasem
from solidspy_uels.solidspy_uels import assem_op_cst, cst_quad9

from time_domain_ccst.cst_utils import assem_op_cst_quad9_rot4, cst_quad9_rot4

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def plot_fields_quad9(bc_array, nodes, elements, solution):
    import matplotlib.pyplot as plt
    import solidspy.postprocesor as pos

    sol_displacement = pos.complete_disp(bc_array, nodes, solution, ndof_node=3)
    pos.plot_node_field(
        sol_displacement[:, 0], nodes, elements, title="X"
    )  # x component
    pos.plot_node_field(
        sol_displacement[:, 1], nodes, elements, title="Y"
    )  # y component
    pos.plot_node_field(
        sol_displacement[:, 2], nodes, elements, title="w"
    )  # y component
    plt.show()


def plot_fields_quad9_rot4(bc_array, nodes, elements, solution):
    import matplotlib.pyplot as plt
    import solidspy.postprocesor as pos

    # # mockup solution vector, just to test constraints
    # for cont in bc_array[:, 0]:
    #     if cont != -1:
    #         solution[cont] = 100
    # for cont in bc_array[:, 1]:
    #     if cont != -1:
    #         solution[cont] = 200
    # for cont in bc_array[:, 2]:
    #     if cont != -1:
    #         solution[cont] = 300

    sol_displacement = pos.complete_disp(bc_array[:, :2], nodes, solution, ndof_node=2)
    pos.plot_node_field(
        sol_displacement[:, 0], nodes, elements, title="X"
    )  # x component
    pos.plot_node_field(
        sol_displacement[:, 1], nodes, elements, title="Y"
    )  # y component

    # x = nodes[:, 1] # in case the other code yields weird stuff
    # y = nodes[:, 2]
    # z = sol_displacement[:, 1]

    # plt.scatter(x, y, c=z, cmap='viridis')
    # plt.colorbar()
    # plt.show()

    vertex_nodes = list(set(elements[:, 3:7].flatten()))
    sol_rotation = pos.complete_disp(
        bc_array[vertex_nodes, 2].reshape(-1, 1),
        nodes[vertex_nodes],
        solution,
        ndof_node=1,
    )

    # this ones doesn't work: (although the solution is well extracted)

    # elements_quad4 = elements[:, :7]
    # elements_quad4[:, 1] = 1
    # for ele in range(elements_quad4.shape[0]):
    #     elements_quad4[ele, 3:] = [np.where(vertex_nodes == elements_quad4[ele, i])[0][0] for i in [3, 4, 5, 6]]
    # pos.plot_node_field(sol_rotation, nodes[vertex_nodes], elements_quad4, title='Rotation')

    x = nodes[vertex_nodes][:, 1]
    y = nodes[vertex_nodes][:, 2]
    z = sol_rotation.flatten()

    fig, ax = plt.subplots()
    # Create a contour plot
    contour = ax.tricontourf(x, y, z, levels=50, cmap="viridis")
    fig.colorbar(contour, ax=ax)
    plt.title("Rotation")
    plt.show()


def retrieve_solution_single_element(params, cst_model):
    cst_model_functions = {
        "cst_quad9_rot4": (assem_op_cst_quad9_rot4, cst_quad9_rot4),
        "cst_quad9": (assem_op_cst, cst_quad9),
    }
    assem_op, cst_element = cst_model_functions[cst_model]
    omega = 1  # TODO: would I need to modify this a lot?
    mats = np.array(
        [
            [
                1.0,
                0.3,
                1.0,
                1.0,
            ]
        ]
    )

    side = params["side"]
    nodes = np.array(
        [
            [0.0, -side / 2, -side / 2],
            [0.0, side / 2, -side / 2],
            [0.0, side / 2, side / 2],
            [0.0, -side / 2, side / 2],
            [0.0, 0.0, -side / 2],
            [0.0, side / 2, 0.0],
            [0.0, 0.0, side / 2],
            [0.0, -side / 2, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    npts = nodes.shape[0]

    # Elements
    elements = np.array([[0, 4, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]])

    cons = np.zeros((npts, 3), dtype=int)
    # lower border is fixed in y and roll in x
    # left border is fixed in x and roll in y
    lower_border = [0, 1, 4]
    left_border = [0, 7, 3]

    upper_border = [3, 6, 2]

    cons[list(lower_border), 1] = -1
    cons[list(left_border), 0] = -1
    loads = np.zeros((npts, 4))  # empty loads
    loads[:, 0] = np.arange(npts)  # specify nodes

    loads[list(upper_border), 1 + 1] = 1  # force in y direction

    assem_op, bc_array, neq = assem_op(cons, elements)
    stiff_mat, mass_mat = assembler(
        elements, mats, nodes, neq, assem_op, uel=cst_element, sparse=False
    )

    rhs = loadasem(loads, bc_array, neq)
    # Solution
    solution = spsolve(stiff_mat - omega**2 * mass_mat, rhs)

    return bc_array, solution, nodes, elements, stiff_mat, mass_mat, rhs


def main():
    params = {"side": 2.0, "mesh_size": 1}

    bc_array, solution, nodes, elements, stiff_mat_4, mass_mat_4, rhs_4  = retrieve_solution_single_element(
        params, "cst_quad9_rot4"
    )

    plot_fields_quad9_rot4(bc_array, nodes, elements, solution)


    bc_array, solution, nodes, elements, stiff_mat_9, mass_mat_9, rhs_9  = retrieve_solution_single_element(
        params, "cst_quad9"
    )

    plot_fields_quad9(bc_array, nodes, elements, solution)

    diff_stiff = stiff_mat_4[0:18, 0:18] - stiff_mat_9[0:18, 0:18]
    diff_mass = mass_mat_4[0:18, 0:18] - mass_mat_9[0:18, 0:18]

    diff_rhs = rhs_4[0:18] - rhs_9[0:18]


if __name__ == "__main__":
    main()
