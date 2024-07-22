from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from solidspy.postprocesor import complete_disp, plot_node_field


def conditional_loads_plotting(
    bc_array,
    nodes,
    rhs,
    elements,
    mesh_size,
    mesh_sizes,
    plot_loads: Literal["all", "last", "none"],
):
    if plot_loads == "all":
        loads = complete_disp(bc_array, nodes, rhs, ndof_node=2)
        plot_node_field(loads, nodes, elements, title=["loads_x", "loads_y "])
    elif plot_loads == "last" and mesh_size == mesh_sizes[-1]:
        loads = complete_disp(bc_array, nodes, rhs, ndof_node=2)
        plot_node_field(loads, nodes, elements, title=["loads_x", "loads_y "])
    else:
        pass


def conditional_fields_plotting(
    u_fem,
    nodes,
    elements,
    u_true,
    mesh_size,
    mesh_sizes,
    plot_field: Literal["all", "last"],
):
    if plot_field == "all":
        plot_node_field(
            u_fem[:, 0],
            nodes,
            elements,
            title=[
                f"u_x FEM_{len(elements)}_elements",
            ],
        )
        plot_node_field(u_true[:, 0], nodes, elements, title=["u_x True"])

    elif plot_field == "last" and mesh_size == mesh_sizes[-1]:
        plot_node_field(
            u_fem[:, 0],
            nodes,
            elements,
            title=[
                f"u_x FEM_{len(elements)}_elements",
            ],
        )
        plot_node_field(u_true[:, 0], nodes, elements, title=["u_x True"])


def convergence_plot(
    mesh_sizes,
    rmses,
):
    log_mesh = np.log10(mesh_sizes)
    log_rmse = np.log10(rmses)

    slope = np.polyfit(log_mesh, log_rmse, 1)[0]

    # and then plot the results
    plt.figure()
    plt.loglog(mesh_sizes, rmses, label="RMSE")
    # plt.loglog(n_elements, max_errors, label="Max Error")
    plt.xlabel("Mesh length")
    plt.ylabel("Error")
    plt.grid()
    plt.legend()

    plt.show()

    print("Slope:", slope)
