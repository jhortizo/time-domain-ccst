import warnings

from time_domain_ccst.fem_solver import retrieve_solution

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def plot_fields(bc_array, nodes, elements, solution):
    import solidspy.postprocesor as pos
    import matplotlib.pyplot as plt

    nnodes = nodes.shape[0]
    nels = elements.shape[0]
    us = solution[:2*nnodes]
    ws = solution[2*nnodes:2*nnodes+nels]
    # ss = solution[-nels:] 

    sol_displacement = pos.complete_disp(bc_array, nodes, us, ndof_node=2)
    pos.plot_node_field(sol_displacement[:, 0], nodes, elements, title='Displacement x')  # x component
    pos.plot_node_field(sol_displacement[:, 1], nodes, elements, title='Displacement y')  # y component

    vertex_nodes = elements[:, 3:7].flatten()
    sol_rotation = pos.complete_disp(bc_array[vertex_nodes, 3], nodes[vertex_nodes], ws, elements[:, :7], 1)

    pos.plot_node_field(sol_rotation, nodes, elements, title='Rotation')

    plt.show()


def main():
    geometry_type = "square"
    params = {"side": 1.0, "mesh_size": 1.0}
    force_reprocess = True

    bc_array, solution, nodes, elements = retrieve_solution(
        geometry_type, params, force_reprocess=force_reprocess
    )

    plot_fields(bc_array, nodes, elements, solution)


if __name__ == "__main__":
    main()
