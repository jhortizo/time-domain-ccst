"""
Contains all the functions to do pretty plots
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import solidspy.postprocesor as pos
from matplotlib.animation import FuncAnimation

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def plot_fields_quad9(bc_array, nodes, elements, solution):
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


def plot_fields_classical(
    bc_array, nodes, elements, solution, instant_show: bool = True
):
    sol_displacement = pos.complete_disp(bc_array, nodes, solution, ndof_node=2)
    pos.plot_node_field(
        sol_displacement[:, 0], nodes, elements, title="X"
    )  # x component
    pos.plot_node_field(
        sol_displacement[:, 1], nodes, elements, title="Y"
    )  # y component
    if instant_show:
        plt.show()


def plot_fields_quad9_rot4(
    bc_array: np.array,
    nodes: np.array,
    elements: np.array,
    solution: np.array,
    instant_show: bool = True,
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
    if instant_show:
        plt.show()


def plot_oscillatory_movement_singleplot(
    x: np.array,
    t: np.array,
    Y: np.array,
    n_plots: int | None = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    savepath: Optional[str] = None,
    instant_show: bool = False,
    colormap: str = "viridis",
) -> None:
    """This requires Y to be of shape (timeframe, values)"""
    if n_plots is None:
        n_plots = len(t)
    time_steps = np.linspace(0, len(t) - 1, n_plots, dtype=int)

    fig, ax = plt.subplots()
    for i in time_steps:
        ax.scatter(x, Y[i, :], c=plt.cm.get_cmap(colormap)(i / len(t)), marker=".", s=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Create a colorbar
    norm = plt.Normalize(vmin=0, vmax=t.max())
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(colormap), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Time")

    if savepath:
        plt.savefig(savepath, dpi=300)

    if instant_show:
        plt.show()


def plot_oscillatory_movement(
    x: np.ndarray,
    t: np.ndarray,
    Y: np.ndarray,
    n_plots: int | None = None,
    savepath: Optional[str] = None,
    fps: int = 1,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
) -> None:
    """This requires Y to be of shape (timeframe, values)"""
    if n_plots is None:
        n_plots = len(t)

    time_steps = np.linspace(0, len(t) - 1, n_plots, dtype=int)

    fig, ax = plt.subplots()
    (line,) = ax.plot(x, Y[time_steps[0]], "k.", markersize=3)

    # be careful, if the norm explodes this will be a problem
    y_range = Y.max() - Y.min()
    y_padding = 0.1 * y_range  # 10% padding
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(Y.min() - y_padding, Y.max() + y_padding)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def update(frame):
        line.set_ydata(Y[time_steps[frame], :])
        time_text.set_text(f"Time: {t[time_steps[frame]]:.2f}")
        return line, time_text

    ani = FuncAnimation(fig, update, frames=len(time_steps), blit=True, interval=50)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ani.save(savepath, fps=fps)


def plot_oscillatory_movement_sample_points(
    solution_displacements: np.ndarray,
    nodes: np.ndarray,
    t: np.ndarray,
    n_points: int = 5,
    savepath: Optional[str] = None,
    instant_show: bool = False,
) -> None:
    """This requires Y to be of shape (timeframe, values)"""

    nodes_x = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), n_points)
    # avoid the first and take the max val
    nodes_x = nodes_x + (nodes_x[1] - nodes_x[0]) # check this toss first and keeps tip

    half_y = nodes[:, 2].max() / 2

    sample_nodes_coordinates = np.array([[x, half_y] for x in nodes_x])
    sample_nodes_ids = np.argmin(
        np.linalg.norm(nodes[:, 1:3] - sample_nodes_coordinates, axis=1), axis=0
    ) # check this retrieves a list of nodes, and the correct ones
    
    sample_solution_displacements = solution_displacements[sample_nodes_ids, :, :]

    sample_disp_norms = np.linalg.norm(sample_solution_displacements, axis=1)

    plt.figure()

    plt.plot(t, sample_disp_norms)
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.legend([f"Node x={round(i, 1)}" for i in nodes_x])

    if savepath:
        plt.savefig(savepath, dpi=300)
    
    if instant_show:
        plt.show()