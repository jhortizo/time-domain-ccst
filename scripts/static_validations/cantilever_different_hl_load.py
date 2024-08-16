import matplotlib.pyplot as plt
import numpy as np
from solidspy.postprocesor import complete_disp
from tqdm import tqdm

from time_domain_ccst.constants import IMAGES_FOLDER
from time_domain_ccst.fem_solver import retrieve_solution


def do_comparison_plotting(
    nodes_lowers: list[np.ndarray],
    norm_u_lowers: list[np.ndarray],
    h_l_ratios: np.ndarray,
) -> None:
    plt.rcParams["font.size"] = 24
    plt.figure(figsize=(12, 7))

    for i in range(len(h_l_ratios)):
        plt.plot(
            nodes_lowers[i][:, 1],
            norm_u_lowers[i, :],
            ".",
            label=f"h/l = {h_l_ratios[i]:.0e}",
        )

    plt.xlabel(r"$x$")
    plt.ylabel("Vertical displacement")
    plt.legend()
    plt.grid()
    plt.savefig(IMAGES_FOLDER + "/compare_cantilever_displacement.png", dpi=300)
    plt.show()


def main():
    geometry_type = "rectangle"
    force_reprocess = False
    cst_model = "cst_quad9_rot4"
    constraints_loads = "cantilever_support_load"

    E = 2
    nu = 0
    rho = 1
    mu = E / (2 * (1 + nu))

    h = 1
    L = 20 * h
    params = {"side_y": h, "side_x": L, "mesh_size": 0.1}

    h_l_ratios = np.logspace(1e-4, 1e4, 9)
    ls = h / h_l_ratios
    etas = ls**2 * mu

    # u_y_lowers = []
    # nodes_lowers = []
    Ks = []
    vertical_load = -1
    cons_loads_fcn_params = {"load": vertical_load}

    for i in tqdm(range(len(h_l_ratios)), desc="h/l ratios"):
        materials = np.array([[E, nu, etas[i], rho]])
        custom_str = f"h_l_ratio_{h_l_ratios[i]:.2e}_load"

        bc_array, solution, nodes, _ = retrieve_solution(
            geometry_type,
            params,
            cst_model,
            constraints_loads,
            materials,
            scenario_to_solve="static",
            force_reprocess=force_reprocess,
            custom_str=custom_str,
            cons_loads_fcn_params=cons_loads_fcn_params,
        )

        u = complete_disp(bc_array, nodes, solution, ndof_node=2)

        # low_border_ids = np.where(nodes[:, 2] == 0)[0]

        # nodes_lower = nodes[low_border_ids]
        # u_lower = u[low_border_ids]

        # nodes_lowers.append(nodes_lower)
        # u_y_lowers.append(u_lower[:, 1])

        # now get the node closest to the tip
        tip_coordinates = np.array([[0, L, h/2]])
        tip_node_id = np.argmin(np.linalg.norm(nodes - tip_coordinates, axis=1))
        tip_y_disp = u[tip_node_id, 1]
        K = abs(vertical_load / tip_y_disp)
        Ks.append(K)

    # max_u_lower_val = abs(np.array(u_y_lowers).min())
    # norm_u_lowers = u_y_lowers / max_u_lower_val
    # do_comparison_plotting(nodes_lowers, norm_u_lowers, h_l_ratios)
    I_ = h**3/12
    adim_stiff = np.array(Ks) * L**3 / (3 * E * I_)

    plt.figure()
    plt.plot(h_l_ratios, adim_stiff, "o-")
    plt.xlabel("h/l")
    plt.ylabel("Stiffness")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()



if __name__ == "__main__":
    main()
