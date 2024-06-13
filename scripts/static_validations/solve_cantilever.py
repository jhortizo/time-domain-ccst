import matplotlib.pyplot as plt
import numpy as np
from solidspy.postprocesor import complete_disp
from tqdm import tqdm

from time_domain_ccst.fem_solver import retrieve_solution


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

    u_lowers = []
    nodes_lowers = []
    for i in tqdm(range(len(h_l_ratios)), desc="h/l ratios"):
        materials = np.array([[E, nu, etas[i], rho]])
        custom_str = f"h_l_ratio_{h_l_ratios[i]:.2e}"

        bc_array, solution, nodes, elements = retrieve_solution(
            geometry_type,
            params,
            cst_model,
            constraints_loads,
            materials,
            force_reprocess=force_reprocess,
            custom_str=custom_str,
        )

        u = complete_disp(bc_array, nodes, solution, ndof_node=2)

        low_border_ids = np.where(nodes[:, 2] == 0)[0]

        nodes_lower = nodes[low_border_ids]
        u_lower = u[low_border_ids]

        nodes_lowers.append(nodes_lower)
        u_lowers.append(u_lower)

    max_u_lower_val = 5e4  # TODO: des machetize this

    norm_u_lowers = [u_lower / max_u_lower_val for u_lower in u_lowers]

    plt.rcParams["font.size"] = 24
    plt.figure(figsize=(12, 6))

    for i in range(len(h_l_ratios)):
        plt.plot(
            nodes_lowers[i][:, 1],
            norm_u_lowers[i][:, 0],
            ".",
            label=f"h/l = {h_l_ratios[i]:.0e}",
        )

    plt.xlabel("x")
    plt.ylabel("Vertical displacement")
    plt.legend()
    plt.grid()
    plt.savefig(
        "compare_cantilever_displacement.png", dpi=300
    )  # TODO: save in proper folder
    plt.show()


if __name__ == "__main__":
    main()
