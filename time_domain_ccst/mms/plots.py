from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from solidspy.postprocesor import complete_disp, plot_node_field
from time_domain_ccst.constants import IMAGES_FOLDER

# rewrites the settings from solidspy.postprocessor
black = "#000000"
plt.rcParams["image.cmap"] = "YlGnBu_r"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["text.color"] = black
plt.rcParams["font.size"] = 12
plt.rcParams["xtick.color"] = black
plt.rcParams["ytick.color"] = black
plt.rcParams["axes.labelcolor"] = black
plt.rcParams["axes.edgecolor"] = black
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False

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
    image_names: str | None = None,
):
    def plot_fields(
        u_fem, nodes, elements, u_true, bc_array, solution, curl_fcn, image_names
    ):
        if image_names:
            savefigs = True
        else:
            savefigs = False

        u_fem_norm = np.linalg.norm(u_fem, axis=1)
        u_true_norm = np.linalg.norm(u_true, axis=1)
        norm_diff = np.abs(u_fem_norm - u_true_norm)

        plot_node_field(
            u_fem_norm,
            nodes,
            elements,
            savefigs=savefigs,
            filename=[
                f"{IMAGES_FOLDER}/{image_names}_u_fem_{len(elements)}_elements.png"
            ],
        )
        plot_node_field(
            u_true_norm,
            nodes,
            elements,
            savefigs=savefigs,
            filename=[
                f"{IMAGES_FOLDER}/{image_names}_u_true_{len(elements)}_elements.png"
            ],
        )
        plot_node_field(
            norm_diff,
            nodes,
            elements,
            savefigs=savefigs,
            filename=[
                f"{IMAGES_FOLDER}/{image_names}_norm_diff_{len(elements)}_elements.png"
            ],
        )

        vertex_nodes = list(set(elements[:, 3:7].flatten()))
        sol_rotation = complete_disp(
            bc_array[vertex_nodes, 2].reshape(-1, 1),
            nodes[vertex_nodes],
            solution,
            ndof_node=1,
        )

        x = nodes[vertex_nodes][:, 1]
        y = nodes[vertex_nodes][:, 2]
        z_fem = sol_rotation.flatten()
        z_teo = curl_fcn(x, y) / 2

        fig, ax = plt.subplots()
        # Create a contour plot
        contour = ax.tricontourf(x, y, z_fem, levels=50, cmap="YlGnBu_r")
        fig.colorbar(contour, ax=ax)
        plt.savefig(
            f"{IMAGES_FOLDER}/{image_names}_curl_fem_{len(elements)}_elements.png",
            dpi=300,
        )

        fig, ax = plt.subplots()
        # Create a contour plot
        contour = ax.tricontourf(x, y, z_teo, levels=50, cmap="YlGnBu_r")
        fig.colorbar(contour, ax=ax)
        plt.savefig(
            f"{IMAGES_FOLDER}/{image_names}_curl_true_{len(elements)}_elements.png",
            dpi=300,
        )

        fig, ax = plt.subplots()
        # Create a contour plot
        contour = ax.tricontourf(
            x, y, np.abs(z_teo - z_fem), levels=50, cmap="YlGnBu_r"
        )
        fig.colorbar(contour, ax=ax)
        plt.savefig(
            f"{IMAGES_FOLDER}/{image_names}_curl_diff_{len(elements)}_elements.png",
            dpi=300,
        )

        # close all figures
        plt.close("all")

    if plot_field == "all":
        plot_fields(
            u_fem, nodes, elements, u_true, bc_array, solution, curl_fcn, image_names
        )
    elif plot_field == "last" and mesh_size == mesh_sizes[-1]:
        plot_fields(
            u_fem, nodes, elements, u_true, bc_array, solution, curl_fcn, image_names
        )


def convergence_plot(
    n_elements,
    errors,
    error_metric_name: str,
    filename: str = None,
):
    log_mesh = np.log10(n_elements)
    log_rmse = np.log10(errors)

    # and then plot the results
    plt.figure()
    plt.loglog(n_elements, errors, "o-", label=error_metric_name)
    # plt.loglog(n_elements, max_errors, label="Max Error")
    plt.xlabel("Number of elements")
    plt.ylabel("Error")
    plt.grid()
    # plt.legend()

    if filename:
        plt.savefig(f"{IMAGES_FOLDER}/{filename}", dpi=300)
    plt.show()

    n_last_points = input(
        "Select number of last points to include in the slope calculation: "
    )
    slope = np.polyfit(log_mesh[-n_last_points:], log_rmse[-n_last_points:], 1)[0]
    print("Slope:", slope)
