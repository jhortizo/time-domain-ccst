from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from solidspy.postprocesor import complete_disp, plot_node_field
from time_domain_ccst.constants import IMAGES_FOLDER


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
    solution,
    bc_array,
    curl_fcn,
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

        vertex_nodes = list(set(elements[:, 3:7].flatten()))
        sol_rotation = complete_disp(
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

        z_teo = curl_fcn(x, y)

        fig, ax = plt.subplots()
        # Create a contour plot
        contour = ax.tricontourf(x, y, z_teo, levels=50, cmap="viridis")
        fig.colorbar(contour, ax=ax)
        plt.title("W_teo")
        plt.show()

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

        vertex_nodes = list(set(elements[:, 3:7].flatten()))
        sol_rotation = complete_disp(
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

        z_teo = curl_fcn(x, y)

        fig, ax = plt.subplots()
        # Create a contour plot
        contour = ax.tricontourf(x, y, z_teo, levels=50, cmap="viridis")
        fig.colorbar(contour, ax=ax)
        plt.title("W_teo")
        plt.show()


def convergence_plot(
    mesh_sizes,
    rmses,
    filename: str = None,
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

    plt.text(0.5, 0.9, f"Slope: {round(slope,2)}", transform=plt.gca().transAxes)
    if filename:
        plt.savefig(f"{IMAGES_FOLDER}/{filename}", dpi=300)
    plt.show()

    print("Slope:", slope)
