import numpy as np
import warnings

from time_domain_ccst.fem_solver import retrieve_solution

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def plot_fields_quad9_rot4(bc_array, nodes, elements, solution):
    import solidspy.postprocesor as pos
    import matplotlib.pyplot as plt

    # # mockup solution vector, just to test constraints
    # for cont in bc_array[:, 0]:
    #     if cont != -1:
    #         solution[cont] = 100
    # for cont in bc_array[:, 1]:
    #     if cont != -1:
    #         solution[cont] = 200
    # for cont in bc_array[:, 2]:
    #     if cont != -1:
    #         solution[cont] = 300

    sol_displacement = pos.complete_disp(bc_array[:, :2], nodes, solution, ndof_node=2)
    pos.plot_node_field(sol_displacement[:, 0], nodes, elements, title='X')  # x component
    pos.plot_node_field(sol_displacement[:, 1], nodes, elements, title='Y')  # y component

    # x = nodes[:, 1] # in case the other code yields weird stuff
    # y = nodes[:, 2]
    # z = sol_displacement[:, 1]

    # plt.scatter(x, y, c=z, cmap='viridis')
    # plt.colorbar()
    # plt.show()
    
    vertex_nodes = list(set(elements[:, 3:7].flatten()))
    sol_rotation = pos.complete_disp(bc_array[vertex_nodes, 2].reshape(-1, 1), nodes[vertex_nodes], solution, ndof_node=1)

    # this ones doesn't work: (although the solution is well extracted)

    # elements_quad4 = elements[:, :7]
    # elements_quad4[:, 1] = 1
    # for ele in range(elements_quad4.shape[0]):
    #     elements_quad4[ele, 3:] = [np.where(vertex_nodes == elements_quad4[ele, i])[0][0] for i in [3, 4, 5, 6]]
    # pos.plot_node_field(sol_rotation, nodes[vertex_nodes], elements_quad4, title='Rotation')

    x = nodes[vertex_nodes][:, 1]
    y = nodes[vertex_nodes][:, 2]
    z = sol_rotation.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, z, cmap='viridis')
    plt.show()


def main():
    geometry_type = "square"
    params = {"side": 1.0, "mesh_size": 1.0}
    force_reprocess = True

    bc_array, solution, nodes, elements = retrieve_solution(
        geometry_type, params, force_reprocess=force_reprocess
    )

    plot_fields_quad9_rot4(bc_array, nodes, elements, solution)


if __name__ == "__main__":
    main()
