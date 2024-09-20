"""
Time-marching scheme is proposed for classical continuum mechanics, taking as
initial state an eigenvector of the system, and without loads.
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.animation import FuncAnimation
from solidspy.postprocesor import complete_disp
from continuum_mechanics import vector


from time_domain_ccst.constants import IMAGES_FOLDER
from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import (
    plot_fields_classical,
    plot_fields_quad9_rot4,
    plot_oscillatory_movement_sample_points_complete_animation_vs_classical,
)

plt.rcParams["image.cmap"] = "YlGnBu_r"
plt.rcParams["mathtext.fontset"] = "cm"


def create_pulse_function(x_length, eta, plotting=False):
    x, y = sp.symbols("x y")

    # gaussian pulse centered at x = 0.5
    u_y = sp.exp(-100 * (x - x_length / 2) ** 2)
    # sp.plotting.plot3d(u_y, (x, 0, 1), (y, 0, 1), title="u_x")
    u_x = 0

    u = sp.Matrix([u_x, u_y, 0])
    w = vector.curl(u) / 2
    # sp.plotting.plot3d(w[2], (x, 0, 2), (y, 0, 1), title="w")

    s = eta / 2 * vector.curl(vector.curl(vector.curl(u)))
    # sp.plotting.plot3d(s[2], (x, 0, 2), (y, 0, 1), title="s")

    initial_state_x = sp.lambdify((x, y), u_x, "numpy")
    initial_state_y = sp.lambdify((x, y), u_y, "numpy")
    initial_state_w = sp.lambdify((x, y), w[2], "numpy")
    initial_state_s = sp.lambdify((x, y), s[2], "numpy")

    initial_state_components = {
        "u_x": initial_state_x,
        "u_y": initial_state_y,
        "w": initial_state_w,
        "s": initial_state_s,
    }

    return initial_state_components


def get_dynamic_solution(
    geometry_type,
    params,
    model,
    constraints_loads,
    materials,
    initial_state,
    custom_str,
    dt,
    n_t_iter,
    force_reprocess,
):
    scenario_to_solve = "time-marching"
    bc_array, solutions, mass_mat, stiff_mat, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        model,
        constraints_loads,
        materials,
        scenario_to_solve=scenario_to_solve,
        force_reprocess=force_reprocess,
        dt=dt,
        n_t_iter=n_t_iter,
        initial_state=initial_state,
        custom_str=custom_str,
        return_matrices=True,
    )

    print("Number of elements:", len(elements))
    # plot_fields_classical(
    #     bc_array,
    #     nodes,
    #     elements,
    #     solutions[:, -1],
    #     instant_show=True,
    # )
    # plot_fields_quad9_rot4(
    #     bc_array,
    #     nodes,
    #     elements,
    #     solutions[:, -1],
    #     instant_show=True,
    # )

    # calculate energy
    d_solutions_u = (solutions[:, 1:] - solutions[:, :-1]) / dt
    energy = []
    for i in range(n_t_iter - 1):
        energy.append(
            solutions[:, i].T @ stiff_mat @ solutions[:, i]
            + d_solutions_u[:, i].T @ mass_mat @ d_solutions_u[:, i]
        )

    solution_displacements = get_displacements(bc_array, nodes, solutions, n_t_iter)

    return solution_displacements, nodes, energy


def get_displacements(bc_array, nodes, solutions, n_iter_t):
    solution_displacements = np.zeros((len(nodes), 2, n_iter_t))
    for i in range(n_iter_t):
        solution_displacements[:, :, i] = complete_disp(
            bc_array[:, :2], nodes, solutions[:, i], ndof_node=2
        )

    return solution_displacements


def plot_and_animation(
    t,
    n_t_iter,
    dt,
    nodes,
    classical_solution_displacements,
    ccst_solution_displacements,
    y_length,
    n_plots=None,
    fps=10,
):
    # -- Fancy plotting

    line_nodes_ids = np.where(np.abs(nodes[:, 2] - y_length / 2) < 0.01)[0]
    line_nodes_ids = line_nodes_ids[np.argsort(nodes[line_nodes_ids, 1])]

    # get u_y values
    cl_u_y = classical_solution_displacements[line_nodes_ids, 1, :]
    ccst_u_y = ccst_solution_displacements[line_nodes_ids, 1, :]

    # static plot
    n_times = 4

    fig, axs = plt.subplots(nrows=n_times, ncols=2, figsize=(6, 6), sharex=True)

    # define some times and find the indices of the closer times to use
    times_to_plot = [0, 1, 1.4, 2.4]
    times_to_plot = [np.argmin(np.abs(t - time)) for time in times_to_plot]

    for i, time in enumerate(times_to_plot):
        axs[i, 0].plot(nodes[line_nodes_ids, 1], ccst_u_y[:, time], "k")
        axs[i, 1].plot(nodes[line_nodes_ids, 1], cl_u_y[:, time], color="gray")

        axs[i, 0].set_ylabel(f"Time: {t[time]:.2f}", x=0.5, y=0.5)

    axs[0, 0].set_title("CCST", x=0.5, y=1.0)
    axs[0, 1].set_title("Classical", x=0.5, y=1.0)

    plt.tight_layout()
    plt.savefig(f"{IMAGES_FOLDER}/pulse_propagation_{dt}.pdf")
    plt.show()

    # animation of section
    n_plots = 500
    if n_plots is None:
        n_plots = len(t)

    time_steps = np.linspace(0, len(t) - 1, n_plots, dtype=int)

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_aspect("equal")

    ax.set_ylim(-1, 1)
    # ax.set_xlim(0, 1)

    (contour_ccst,) = ax.plot(
        nodes[line_nodes_ids, 1],
        ccst_u_y[:, time_steps[0]],
        "k",
    )
    (contour_classical,) = ax.plot(
        nodes[line_nodes_ids, 1],
        cl_u_y[:, time_steps[0]],
        color="gray",
        linestyle="--",
    )

    time_text = ax.text(0.02, 1, f"Time: {t[0]:.2f}", transform=ax.transAxes)

    def update(frame):
        contour_ccst.set_ydata(ccst_u_y[:, time_steps[frame]])
        contour_classical.set_ydata(cl_u_y[:, time_steps[frame]])
        time_text.set_text(f"Time: {t[time_steps[frame]]:.2f}")

        return (
            contour_ccst,
            contour_classical,
            time_text,
        )

    ani = FuncAnimation(fig, update, frames=n_plots, blit=True, interval=50)

    ani.save(
        f"{IMAGES_FOLDER}/pulse_propagation_{dt}.gif",
        fps=fps,
    )


def main():
    # -- Different cases run in this script

    t_final = 3.5
    dt = 0.0001
    n_t_iter = int(t_final / dt)

    # -- Overall constants
    geometry_type = "rectangle"
    x_length = 1.5
    y_length = 0.3
    params = {"side_x": x_length, "side_y": y_length, "mesh_size": 0.1}
    force_reprocess = False
    plotting = False

    ccst_constraints_loads = "pulse_ccst"
    cst_model = "cst_quad9_rot4"
    ccst_materials = np.array(
        [
            [
                1,  # E, young's modulus
                0.29,  # nu, poisson's ratio
                0.001,  # eta, coupling parameter
                1,  # rho, density
            ]
        ]
    )

    classical_constraints_loads = "pulse_classical"
    classical_model = "classical_quad9"
    classical_materials = np.array(
        [
            [
                1,  # E, young's modulus
                0.29,  # nu, poisson's ratio
                1,  # rho, density
            ]
        ]
    )

    # -- Finding initial states
    common_initial_state = create_pulse_function(
        x_length, ccst_materials[0, 2], plotting
    )

    # -- Getting dynamic solutions
    custom_str = f"pulse_n_t_iter_{n_t_iter}_dt_{dt}_eta_{ccst_materials[0, 2]}_ccst"
    ccst_solution_displacements, nodes, ccst_energy = get_dynamic_solution(
        geometry_type,
        params,
        cst_model,
        ccst_constraints_loads,
        ccst_materials,
        common_initial_state,
        custom_str,
        dt,
        n_t_iter,
        force_reprocess,
    )

    custom_str = (
        f"pulse_n_t_iter_{n_t_iter}_dt_{dt}_eta_{ccst_materials[0, 2]}_classical"
    )
    classical_solution_displacements, nodes, classical_energy = get_dynamic_solution(
        geometry_type,
        params,
        classical_model,
        classical_constraints_loads,
        classical_materials,
        common_initial_state,
        custom_str,
        dt,
        n_t_iter,
        force_reprocess,
    )

    t = np.linspace(0, n_t_iter * dt, n_t_iter)

    plot_and_animation(
        t,
        n_t_iter,
        dt,
        nodes,
        classical_solution_displacements,
        ccst_solution_displacements,
        y_length,
    )

    t = t[:-1]
    plt.figure(figsize=(8, 3))
    plt.plot(
        t,
        classical_energy / classical_energy[0],
        label="Classical",
        color="gray",
        linestyle="--",
    )

    plt.plot(
        t, ccst_energy / ccst_energy[0], label="CCST", color="black", linestyle="-"
    )

    plt.xlabel(r"$t$")
    plt.ylabel(r"$E / E_0$")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_FOLDER}/pulse_energy_comparison_{dt}.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
