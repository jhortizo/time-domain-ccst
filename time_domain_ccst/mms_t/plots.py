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
    u_fems,
    u_trues,
    plot_field: Literal["all", "last"],
    mesh_size,
    mesh_sizes,
    n_points=3,
):
    def plot_fields(u_fems, u_trues, n_points):
        random_generator = np.random.default_rng(42)
        ids_to_plot = random_generator.choice(u_fems.shape[0], 10, replace=False)

        u_fem_to_plot = u_fems[ids_to_plot, :, :]
        u_fem_to_plot = np.linalg.norm(u_fem_to_plot, axis=1)

        u_true_to_plot = u_trues[ids_to_plot, :, :]
        u_true_to_plot = np.linalg.norm(u_true_to_plot, axis=1)

        plt.figure()
        plt.plot(u_fem_to_plot.T, 'b', label="FEM")
        plt.plot(u_true_to_plot.T, 'k--', label="True")
        plt.legend()

        plt.show()

    if plot_field == "all":
        plot_fields(u_fems, u_trues, n_points=n_points)
    elif plot_field == "last" and mesh_size == mesh_sizes[-1]:
        plot_fields(u_fems, u_trues, n_points=n_points)


def convergence_plot(
    mesh_sizes,
    errors,
    error_metric_name: str,
    filename: str = None,
):
    log_mesh = np.log10(mesh_sizes)
    log_rmse = np.log10(errors)

    slope = np.polyfit(log_mesh, log_rmse, 1)[0]

    # and then plot the results
    plt.figure()
    plt.loglog(mesh_sizes, errors, "o-", label=error_metric_name)
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
