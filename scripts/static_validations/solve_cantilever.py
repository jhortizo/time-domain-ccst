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

    E = 1
    nu = 0
    rho = 1
    mu = E / (2 * (1 + nu))

    h = 1
    L = 20 * h
    params = {"side_y": h, "side_x": L, "mesh_size": 0.1}

    h_l_ratios = np.array([1e4, 2, 1e-4])
    ls = h / h_l_ratios
    etas = ls**2 * mu

    u_y_lowers = []
    nodes_lowers = []
    for i in tqdm(range(len(h_l_ratios)), desc="h/l ratios"):
        materials = np.array([[E, nu, etas[i], rho]])
        custom_str = f"h_l_ratio_{h_l_ratios[i]:.2e}"

        bc_array, solution, nodes, _ = retrieve_solution(
            geometry_type,
            params,
            cst_model,
            constraints_loads,
            materials,
            scenario_to_solve='static',
            force_reprocess=force_reprocess,
            custom_str=custom_str,
        )

        u = complete_disp(bc_array, nodes, solution, ndof_node=2)

        low_border_ids = np.where(nodes[:, 2] == 0)[0]

        nodes_lower = nodes[low_border_ids]
        u_lower = u[low_border_ids]

        nodes_lowers.append(nodes_lower)
        u_y_lowers.append(u_lower[:, 1])

    max_u_lower_val = abs(np.array(u_y_lowers).min())
    norm_u_lowers = u_y_lowers / max_u_lower_val
    do_comparison_plotting(nodes_lowers, norm_u_lowers, h_l_ratios)


if __name__ == "__main__":
    main()
