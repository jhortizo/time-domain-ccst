"""
Time-marching scheme is proposed for classical continuum mechanics, taking as
initial state an eigenvector of the system, and without loads.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from solidspy.postprocesor import complete_disp

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import plot_fields_classical


def animate_bottom_line(bc_array, nodes, solutions, n_iter_t):
    # get the indices of the bottom line nodes
    lower_border_ids = np.where(nodes[:, 2] == 0)[0]

    # calculate the displacement solution for each time step
    solution_displacements = np.zeros((len(nodes), 2, n_iter_t))
    for i in range(n_iter_t):
        solution_displacements[:, :, i] = complete_disp(
            bc_array, nodes, solutions[:, i], ndof_node=2
        )

    # get the displacement solution for the bottom line nodes
    lower_border_y_displacement = solution_displacements[lower_border_ids, 1, :]

    # do initial plotting, just the x values of the nodes vs the y displacement
    fig, ax = plt.subplots()
    lower_border_plot = ax.plot(
        nodes[lower_border_ids, 1], lower_border_y_displacement[:, 0], 'k.'
    )[0]

    def update_plot(i):
        lower_border_plot.set_ydata(lower_border_y_displacement[:, i])
        return lower_border_plot, 

    # use the function to create the animation plot
    lower_border_animation = FuncAnimation(fig, update_plot, frames=range(n_iter_t), blit=True, interval=1000)
    lower_border_animation.save("lower_border_animation.gif", writer="pillow")
    


def main():
    geometry_type = "rectangle"
    params = {"side_x": 10.0, "side_y": 1.0, "mesh_size": 1.0}
    force_reprocess = True
    cst_model = "classical_quad9"
    constraints_loads = "cantilever_support_classical"

    materials = np.array(
        [
            [
                1,  # E, young's modulus
                0.29,  # nu, poisson's ratio
                1,  # rho, density
            ]
        ]
    )

    # first solve the eigenvalue problem and acquire an eigenstate
    scenario_to_solve = "eigenproblem"

    bc_array, eigvals, eigvecs, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        materials,
        scenario_to_solve="eigenproblem",
        force_reprocess=force_reprocess,
    )

    n_eigvec = 0
    # plot_fields_classical(bc_array, nodes, elements, eigvecs[:, n_eigvec], instant_show=False)

    # the mesh and constraints are the same, so the exact structure of the eigvecs array
    # can be used as initial state

    initial_state = eigvecs[:, n_eigvec]
    scenario_to_solve = "time-marching"
    n_t_iter = 1000
    dt = 0.01
    bc_array, solutions, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        materials,
        scenario_to_solve=scenario_to_solve,
        force_reprocess=force_reprocess,
        dt=dt,
        n_t_iter=n_t_iter,
        initial_state=initial_state,
    )

    # for i in range(10):
    #     plot_fields_classical(bc_array, nodes, elements, solutions[:, i])
    # plt.show()

    animate_bottom_line(bc_array, nodes, solutions, n_t_iter)


if __name__ == "__main__":
    main()
