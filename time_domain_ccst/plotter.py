"""
Contains all the functions to do pretty plots
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import solidspy.postprocesor as pos
from matplotlib.animation import FuncAnimation

from .constants import IMAGES_FOLDER

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


def plot_oscillatory_movement_sample_points_complete_animation(
    solution_displacements: np.ndarray,
    nodes: np.ndarray,
    t: np.ndarray,
    n_points: int = 5,
    fps: int = 10,
    n_plots: int | None = None,
    custom_str: Optional[str] = None,
    instant_show: bool = False,
) -> None:

    nodes_x = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), n_points + 1)[1:]

    half_y = nodes[:, 2].max() / 2

    sample_nodes_coordinates = np.array([[x, half_y] for x in nodes_x])
    sample_nodes_ids = [
        np.argmin(
            np.linalg.norm(nodes[:, 1:] - sample_nodes_coordinates[i, :], axis=1),
            axis=0,
        )
        for i in range(n_points)
    ]

    sample_solution_displacements = solution_displacements[sample_nodes_ids, :, :]
    all_nodes_positions = np.zeros_like(solution_displacements)
    normalized_displacements = solution_displacements / solution_displacements.max()
    for i in range(all_nodes_positions.shape[2]):
        all_nodes_positions[:, :, i] = nodes[:, 1:] + normalized_displacements[:, :, i]

    # get border nodes
    bottom_border_nodes = np.where(nodes[:, 2] == nodes[:, 2].min())[0]
    top_border_nodes = np.where(nodes[:, 2] == nodes[:, 2].max())[0]
    left_border_nodes = np.where(nodes[:, 1] == nodes[:, 1].min())[0]
    right_border_nodes = np.where(nodes[:, 1] == nodes[:, 1].max())[0]

    # organize borders so a line plot is organized, clockwise
    bottom_border_nodes = bottom_border_nodes[np.argsort(nodes[bottom_border_nodes, 1])]
    right_border_nodes = right_border_nodes[np.argsort(nodes[right_border_nodes, 2])]
    top_border_nodes = top_border_nodes[np.argsort(nodes[top_border_nodes, 1])][::-1]
    left_border_nodes = left_border_nodes[np.argsort(nodes[left_border_nodes, 2])][::-1]

    border_nodes = np.concatenate(
        [
            bottom_border_nodes,
            right_border_nodes,
            top_border_nodes,
            left_border_nodes,
        ]
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    ax1.plot(
        all_nodes_positions[border_nodes, 0, 0],
        all_nodes_positions[border_nodes, 1, 0],
        "k",
    )
    for i in range(n_points):
        ax1.plot(
            all_nodes_positions[sample_nodes_ids[i], 0, 0],
            all_nodes_positions[sample_nodes_ids[i], 1, 0],
            "o",
        )
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_aspect("equal")
    ax1.axis("off")

    for i in range(n_points):
        ax2.plot(
            t,
            sample_solution_displacements[i, 1, :],
            label=f"x={round(nodes_x[i], 1)}",
        )
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$u_y$")
    ax2.set_aspect("auto")
    plt.tight_layout()

    if custom_str:
        plt.savefig(
            IMAGES_FOLDER
            + f"/ccst_fixed_cantilever_{custom_str}_implicit_sample_points.png",
            dpi=300,
        )

    if instant_show:
        plt.show()

    # and then it comes the animation, complete
    if n_plots is None:
        n_plots = len(t)

    time_steps = np.linspace(0, len(t) - 1, n_plots, dtype=int)

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_ylim(all_nodes_positions[:, 1, :].min(), all_nodes_positions[:, 1, :].max())
    ax.set_xlim(all_nodes_positions[:, 0, :].min(), all_nodes_positions[:, 0, :].max())

    (contour,) = ax.plot(
        all_nodes_positions[border_nodes, 0, time_steps[0]],
        all_nodes_positions[border_nodes, 1, time_steps[0]],
        "k",
    )

    points = []

    for i in range(n_points):
        (point,) = ax.plot(
            all_nodes_positions[sample_nodes_ids[i], 0, time_steps[0]],
            all_nodes_positions[sample_nodes_ids[i], 1, time_steps[0]],
            "o",
        )
        points.append(point)

    time_text = ax.text(0.02, 1, f"Time: {t[0]:.2f}", transform=ax.transAxes)

    def update(frame):
        contour.set_xdata(all_nodes_positions[border_nodes, 0, time_steps[frame]])
        contour.set_ydata(all_nodes_positions[border_nodes, 1, time_steps[frame]])
        time_text.set_text(f"Time: {t[time_steps[frame]]:.2f}")
        for i in range(n_points):
            points[i].set_xdata(
                all_nodes_positions[sample_nodes_ids[i], 0, time_steps[frame]]
            )
            points[i].set_ydata(
                all_nodes_positions[sample_nodes_ids[i], 1, time_steps[frame]]
            )
        return contour, *points, time_text

    ani = FuncAnimation(fig, update, frames=len(time_steps), blit=True, interval=50)

    ani.save(
        IMAGES_FOLDER + f"/ccst_fixed_cantilever_{custom_str}_implicit_complete.gif",
        fps=fps,
    )
