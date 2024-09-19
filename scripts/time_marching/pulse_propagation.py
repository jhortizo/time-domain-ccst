"""
Time-marching scheme is proposed for classical continuum mechanics, taking as
initial state an eigenvector of the system, and without loads.
"""

import matplotlib.pyplot as plt
import numpy as np
from solidspy.postprocesor import complete_disp

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import (
    plot_fields_classical,
    plot_fields_quad9_rot4,
    plot_oscillatory_movement_sample_points_complete_animation_vs_classical,
)

plt.rcParams["image.cmap"] = "YlGnBu_r"
plt.rcParams["mathtext.fontset"] = "cm"


def create_pulse_function(plotting=False):
    import sympy as sp

    x, y = sp.symbols("x y")

    # gaussian pulse centered at x = 0.5
    u_y = sp.exp(-100 * (x - 0.5) ** 2)
    # sp.plotting.plot3d(u_y, (x, 0, 1), (y, 0, 1), title="u_x")
    u_x = 0

    initial_state_x = sp.lambdify((x, y), u_x, "numpy")
    initial_state_y = sp.lambdify((x, y), u_y, "numpy")

    initial_state_components = {
        "u_x": initial_state_x,
        "u_y": initial_state_y,
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
    bc_array, solutions, nodes, elements = retrieve_solution(
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
    )

    plot_fields_classical(
        bc_array,
        nodes,
        elements,
        solutions[:, 100],
        instant_show=True,
    )

    solution_displacements = get_displacements(bc_array, nodes, solutions, n_t_iter)

    return solution_displacements, nodes


def get_displacements(bc_array, nodes, solutions, n_iter_t):
    solution_displacements = np.zeros((len(nodes), 2, n_iter_t))
    for i in range(n_iter_t):
        solution_displacements[:, :, i] = complete_disp(
            bc_array[:, :2], nodes, solutions[:, i], ndof_node=2
        )

    return solution_displacements


def main():
    # -- Different cases run in this script

    dt = 0.01

    # -- Overall constants
    geometry_type = "rectangle"
    params = {"side_x": 1.0, "side_y": 1.0, "mesh_size": 0.25}
    force_reprocess = False
    plotting = False

    ccst_constraints_loads = "cantilever_support"
    cst_model = "cst_quad9_rot4"
    ccst_materials = np.array(
        [
            [
                1,  # E, young's modulus
                0.29,  # nu, poisson's ratio
                0.1,  # eta, coupling parameter
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

    n_t_iter = 1000

    # -- Finding initial states
    common_initial_state = create_pulse_function(plotting)

    # -- Getting dynamic solutions
    # custom_str = f"pulse_n_t_iter_{n_t_iter}_dt_{dt}_eta_{ccst_materials[0, 2]}_ccst"
    # ccst_solution_displacements, nodes = get_dynamic_solution(
    #     geometry_type,
    #     params,
    #     cst_model,
    #     ccst_constraints_loads,
    #     ccst_materials,
    #     common_initial_state,
    #     custom_str,
    #     dt,
    #     n_t_iter,
    #     force_reprocess,
    # )

    custom_str = (
        f"pules_n_t_iter_{n_t_iter}_dt_{dt}_eta_{ccst_materials[0, 2]}_classical"
    )
    classical_solution_displacements, nodes = get_dynamic_solution(
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

    # -- Fancy plotting
    ts = np.linspace(0, n_t_iter * dt, n_t_iter)

    ccst_solution_displacements = classical_solution_displacements

    line_nodes_ids = np.where(np.abs(nodes[:, 2] - 0.5) < 0.01)[0]
    line_nodes_ids = line_nodes_ids[np.argsort(nodes[line_nodes_ids, 1])]

    # get u_y values
    cl_u_y = classical_solution_displacements[line_nodes_ids, 1, :]
    ccst_u_y = ccst_solution_displacements[line_nodes_ids, 1, :]

    n_plots = 1000
    t = ts
    # and then it comes the animation, complete
    if n_plots is None:
        n_plots = len(t)

    time_steps = np.linspace(0, len(t) - 1, n_plots, dtype=int)

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_aspect("equal")

    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 1)

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

    from matplotlib.animation import FuncAnimation

    ani = FuncAnimation(fig, update, frames=len(time_steps), blit=True, interval=50)

    fps = 10
    ani.save(
        "pulse_propagation.gif",
        fps=fps,
    )


if __name__ == "__main__":
    main()
