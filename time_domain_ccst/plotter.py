"""
Contains all the functions to do pretty plots
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
import solidspy.postprocesor as pos

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def plot_fields(bc_array, nodes, elements, solution):
    
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


def plot_fields_quad9_rot4(
    bc_array: np.array, nodes: np.array, elements: np.array, solution: np.array
) -> None:

    sol_displacement = pos.complete_disp(bc_array[:, :2], nodes, solution, ndof_node=2)
    pos.plot_node_field(
        sol_displacement[:, 0], nodes, elements, title="X"
    )  # x component
    pos.plot_node_field(
        sol_displacement[:, 1], nodes, elements, title="Y"
    )  # y component

    # and plot the rotations field
    vertex_nodes = list(set(elements[:, 3:7].flatten()))
    sol_rotation = pos.complete_disp(
        bc_array[vertex_nodes, 2].reshape(-1, 1),
        nodes[vertex_nodes],
        solution,
        ndof_node=1,
    )

    x = nodes[vertex_nodes][:, 1]
    y = nodes[vertex_nodes][:, 2]
    z = sol_rotation.flatten()

    fig, ax = plt.subplots()
    # Create a contour plot
    contour = ax.tricontourf(x, y, z, levels=50, cmap="viridis")
    fig.colorbar(contour, ax=ax)
    plt.title("W")
    plt.show()
