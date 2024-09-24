"""
Contains all the functions to do pretty plots
"""

import warnings
from typing import Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
import solidspy.postprocesor as pos
from matplotlib.animation import FuncAnimation

from .constants import IMAGES_FOLDER
from matplotlib.gridspec import GridSpec

plt.style.use("cst_paper.mplstyle")

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
    n_points: int = 3,
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
    normalized_displacements = (
        0.5 * solution_displacements / np.abs(solution_displacements).max()
    )
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

    fig = plt.figure(layout="constrained", figsize=(8, 3))
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax4 = fig.add_subplot(gs[2, 1], sharex=ax2)

    axs = [ax2, ax3, ax4]
    ax1.plot(
        all_nodes_positions[border_nodes, 0, 0],
        all_nodes_positions[border_nodes, 1, 0],
        "gray",
    )

    for i in range(n_points):
        ax1.plot(
            all_nodes_positions[sample_nodes_ids[i], 0, 0],
            all_nodes_positions[sample_nodes_ids[i], 1, 0],
            "o",
            color="black",
        )

    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_aspect("equal")
    ax1.axis("off")

    for i in range(n_points):
        yvals = sample_solution_displacements[i, 1, :]

        axs[i].set_ylabel(r"$u_y$")

        axs[i].plot(
            t,
            yvals,
            label=f"x={round(nodes_x[i], 1)}",
            color="black",
        )

        axs[i].xaxis.set_visible(False)
        axs[i].set_aspect("auto")
        axs[i].set_yticks([])

    ax4.xaxis.set_visible(True)
    ax4.set_xlabel(r"$t$")
    plt.tight_layout()

    if custom_str:
        plt.savefig(
            IMAGES_FOLDER
            + f"/ccst_fixed_cantilever_{custom_str}_implicit_sample_points.pdf",
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
            color="black",
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


def plot_oscillatory_movement_sample_points_complete_animation_vs_classical(
    ccst_solution_displacements: np.ndarray,
    classical_solution_displacements: np.ndarray,
    nodes: np.ndarray,
    t: np.ndarray,
    n_points: int = 5,
    fps: int = 10,
    n_plots: int | None = None,
    custom_str: Optional[str] = None,
    instant_show: bool = False,
    static_field_to_plot: Literal["x", "y", "norm"] = "y",
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

    sample_solution_displacements_ccst = ccst_solution_displacements[
        sample_nodes_ids, :, :
    ]
    sample_solution_displacements_classical = classical_solution_displacements[
        sample_nodes_ids, :, :
    ]
    all_nodes_positions_ccst = np.zeros_like(ccst_solution_displacements)
    all_nodes_positions_classical = np.zeros_like(classical_solution_displacements)
    normalized_displacements_ccst = (
        ccst_solution_displacements / np.abs(ccst_solution_displacements).max()
    )
    normalized_displacements_classical = (
        classical_solution_displacements
        / np.abs(classical_solution_displacements).max()
    )
    for i in range(all_nodes_positions_ccst.shape[2]):
        all_nodes_positions_ccst[:, :, i] = (
            nodes[:, 1:] + normalized_displacements_ccst[:, :, i]
        )
        all_nodes_positions_classical[:, :, i] = (
            nodes[:, 1:] + normalized_displacements_classical[:, :, i]
        )

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

    fig = plt.figure(layout="constrained", figsize=(8, 3))
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
    ax4 = fig.add_subplot(gs[2, 1], sharex=ax2)

    axs = [ax2, ax3, ax4]
    ax1.plot(
        all_nodes_positions_ccst[border_nodes, 0, 0],
        all_nodes_positions_ccst[border_nodes, 1, 0],
        "gray",
    )
    ax1.plot(
        all_nodes_positions_classical[border_nodes, 0, 0],
        all_nodes_positions_classical[border_nodes, 1, 0],
        color="black",
        linestyle=":",
    )

    for i in range(n_points):
        ax1.plot(
            all_nodes_positions_ccst[sample_nodes_ids[i], 0, 0],
            all_nodes_positions_ccst[sample_nodes_ids[i], 1, 0],
            "o",
            color="black",
        )
        ax1.plot(
            all_nodes_positions_classical[sample_nodes_ids[i], 0, 0],
            all_nodes_positions_classical[sample_nodes_ids[i], 1, 0],
            "d",
            color="black",
        )
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_aspect("equal")
    ax1.axis("off")

    for i in range(n_points):
        if static_field_to_plot == "y":
            yvals_ccst = sample_solution_displacements_ccst[i, 1, :]
            yvals_classical = sample_solution_displacements_classical[i, 1, :]
            axs[i].set_ylabel(r"$u_y$")
        elif static_field_to_plot == "x":
            yvals_ccst = sample_solution_displacements_ccst[i, 0, :]
            yvals_classical = sample_solution_displacements_classical[i, 0, :]
            axs[i].set_ylabel(r"$u_x$")
        elif static_field_to_plot == "norm":
            yvals_ccst = np.linalg.norm(
                sample_solution_displacements_ccst[i, :, :], axis=0
            )
            yvals_classical = np.linalg.norm(
                sample_solution_displacements_classical[i, :, :], axis=0
            )
            axs[i].set_ylabel(r"$||u||$")

        axs[i].plot(
            t,
            yvals_ccst,
            label=f"x={round(nodes_x[i], 1)}",
            color="black",
        )
        axs[i].plot(
            t,
            yvals_classical,
            "--",
            color="black",
        )

        axs[i].set_aspect("auto")
        axs[i].xaxis.set_visible(False)
        axs[i].set_yticks([])

    ax4.xaxis.set_visible(True)
    ax4.set_xlabel(r"$t$")
    plt.tight_layout()

    if custom_str:
        plt.savefig(
            IMAGES_FOLDER
            + f"/ccst_fixed_cantilever_{custom_str}_implicit_sample_points_vs_classical_{static_field_to_plot}.pdf",
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

    overall_min_y = min(
        all_nodes_positions_ccst[:, 1, :].min(),
        all_nodes_positions_classical[:, 1, :].min(),
    )

    overall_max_y = max(
        all_nodes_positions_ccst[:, 1, :].max(),
        all_nodes_positions_classical[:, 1, :].max(),
    )

    overall_min_x = min(
        all_nodes_positions_ccst[:, 0, :].min(),
        all_nodes_positions_classical[:, 0, :].min(),
    )

    overall_max_x = max(
        all_nodes_positions_ccst[:, 0, :].max(),
        all_nodes_positions_classical[:, 0, :].max(),
    )

    ax.set_ylim(overall_min_y, overall_max_y)
    ax.set_xlim(overall_min_x, overall_max_x)

    (contour_ccst,) = ax.plot(
        all_nodes_positions_ccst[border_nodes, 0, time_steps[0]],
        all_nodes_positions_ccst[border_nodes, 1, time_steps[0]],
        "k",
    )
    (contour_classical,) = ax.plot(
        all_nodes_positions_classical[border_nodes, 0, time_steps[0]],
        all_nodes_positions_classical[border_nodes, 1, time_steps[0]],
        color="gray",
        linestyle="--",
    )

    points_ccst = []
    points_classical = []

    for i in range(n_points):
        (point_ccst,) = ax.plot(
            all_nodes_positions_ccst[sample_nodes_ids[i], 0, time_steps[0]],
            all_nodes_positions_ccst[sample_nodes_ids[i], 1, time_steps[0]],
            "o",
            color="black",
        )
        (point_classical,) = ax.plot(
            all_nodes_positions_classical[sample_nodes_ids[i], 0, time_steps[0]],
            all_nodes_positions_classical[sample_nodes_ids[i], 1, time_steps[0]],
            "d",
            color="black",
        )
        points_ccst.append(point_ccst)
        points_classical.append(point_classical)

    time_text = ax.text(0.02, 1, f"Time: {t[0]:.2f}", transform=ax.transAxes)

    def update(frame):
        contour_ccst.set_xdata(
            all_nodes_positions_ccst[border_nodes, 0, time_steps[frame]]
        )
        contour_ccst.set_ydata(
            all_nodes_positions_ccst[border_nodes, 1, time_steps[frame]]
        )
        contour_classical.set_xdata(
            all_nodes_positions_classical[border_nodes, 0, time_steps[frame]]
        )
        contour_classical.set_ydata(
            all_nodes_positions_classical[border_nodes, 1, time_steps[frame]]
        )
        time_text.set_text(f"Time: {t[time_steps[frame]]:.2f}")
        for i in range(n_points):
            points_ccst[i].set_xdata(
                all_nodes_positions_ccst[sample_nodes_ids[i], 0, time_steps[frame]]
            )
            points_ccst[i].set_ydata(
                all_nodes_positions_ccst[sample_nodes_ids[i], 1, time_steps[frame]]
            )
            points_classical[i].set_xdata(
                all_nodes_positions_classical[sample_nodes_ids[i], 0, time_steps[frame]]
            )
            points_classical[i].set_ydata(
                all_nodes_positions_classical[sample_nodes_ids[i], 1, time_steps[frame]]
            )
        return (
            contour_ccst,
            contour_classical,
            *points_ccst,
            *points_classical,
            time_text,
        )

    ani = FuncAnimation(fig, update, frames=len(time_steps), blit=True, interval=50)

    ani.save(
        IMAGES_FOLDER
        + f"/ccst_fixed_cantilever_{custom_str}_implicit_complete_vs_classical.gif",
        fps=fps,
    )
