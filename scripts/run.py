import warnings

from time_domain_ccst.fem_solver import retrieve_solution

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def plot_field(bc_array, nodes, us, elements):
    import solidspy.postprocesor as pos
    import matplotlib.pyplot as plt

    sol = pos.complete_disp(bc_array, nodes, us, ndof_node=2)
    pos.plot_node_field(sol[:, 0], nodes, elements)  # x component
    pos.plot_node_field(sol[:, 1], nodes, elements)  # y component

    # TODO: rotations field is missing

    plt.show()


def main():
    geometry_type = "square"
    params = {"side": 1.0, "mesh_size": 1.0}
    force_reprocess = True

    bc_array, solution, nodes, elements = retrieve_solution(
        geometry_type, params, force_reprocess=force_reprocess
    )

    nnodes = nodes.shape[0]
    nels = elements.shape[0]
    us = solution[:2*nnodes]
    ws = solution[2*nnodes:2*nnodes+nels]
    ss = solution[-nels:] 

    plot_field(bc_array, nodes, us, elements)


if __name__ == "__main__":
    main()
