import matplotlib.pyplot as plt
import numpy as np
from solidspy.postprocesor import complete_disp
from tqdm import tqdm

from time_domain_ccst.constants import IMAGES_FOLDER
from time_domain_ccst.fem_solver import retrieve_solution


def do_stiffness_variation_plotting(
    h_l_ratios: np.ndarray,
    adim_stiff: np.ndarray,
) -> None:
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(12, 7))

    plt.plot(h_l_ratios, adim_stiff, "o-")
    plt.xlabel("h/l")
    plt.ylabel("Stiffness")
    plt.xscale("log")
    plt.yscale("log")

    plt.savefig(IMAGES_FOLDER + "/compare_rigidity_variation.png", dpi=300)
    plt.show()


def main():
    geometry_type = "rectangle"
    force_reprocess = True
    cst_model = "cst_quad9_rot4"
    constraints_loads = "cantilever_support_load"

    E = 2
    nu = 0
    rho = 1
    mu = E / (2 * (1 + nu))

    h = 1
    L = 20 * h
    params = {"side_y": h, "side_x": L, "mesh_size": 0.1}

    h_l_ratios = np.logspace(-4, 4, 9)
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

        tip_coordinates = np.array([[0, L, h / 2]])
        tip_node_id = np.argmin(np.linalg.norm(nodes - tip_coordinates, axis=1))
        tip_y_disp = u[tip_node_id, 1]
        K = abs(vertical_load / tip_y_disp)
        Ks.append(K)

    I_ = h**3 / 12
    adim_stiff = np.array(Ks) * L**3 / (3 * E * I_)
    do_stiffness_variation_plotting(h_l_ratios, adim_stiff)


if __name__ == "__main__":
    main()
