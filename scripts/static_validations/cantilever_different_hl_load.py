import matplotlib.pyplot as plt
import numpy as np
from solidspy.postprocesor import complete_disp
from tqdm import tqdm

from time_domain_ccst.constants import IMAGES_FOLDER
from time_domain_ccst.fem_solver import retrieve_solution

plt.style.use("cst_paper.mplstyle")


def do_stiffness_variation_plotting(
    h_l_ratios: np.ndarray, adim_stiffs: np.ndarray, L_names: list[str]
) -> None:
    plt.rcParams["image.cmap"] = "YlGnBu_r"
    plt.rcParams["mathtext.fontset"] = "cm"

    for admin_stiff, L_name in zip(adim_stiffs, L_names):
        plt.plot(h_l_ratios, admin_stiff, "o-", label=f"$L= {L_name}h$")
    plt.xlabel(r"Characteristic Geometry Ratio $h/l$")
    plt.ylabel(r"Nondimensional Stiffness $K L^3 / (3 E I)$")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.legend()

    plt.savefig(IMAGES_FOLDER + "/compare_rigidity_variation.pdf", dpi=300)
    plt.show()


def main():
    geometry_type = "rectangle"
    force_reprocess = False
    cst_model = "cst_quad9_rot4"
    constraints_loads = "cantilever_support_load"
    vertical_load = -1
    cons_loads_fcn_params = {"load": vertical_load}

    E = 2
    nu = 0
    rho = 1
    mu = E / (2 * (1 + nu))

    h = 1
    L_factors = [20, 40]
    Ls = [factor * h for factor in L_factors]
    L_names = [str(factor) for factor in L_factors]

    h_l_ratios = np.logspace(-4, 4, 17)
    ls = h / h_l_ratios
    etas = ls**2 * mu

    Kss = []

    # takes about 15min in my machine
    for i in tqdm(range(len(h_l_ratios)), desc="h/l ratios"):
        Ks = []
        for L in Ls:
            params = {"side_y": h, "side_x": L, "mesh_size": 0.1}
            materials = np.array([[E, nu, etas[i], rho]])
            custom_str = f"h_l_ratio_{h_l_ratios[i]:.2e}_L_h_ratio_{L:.2e}"

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
        Kss.append(Ks)

    Ks = np.array(Kss)
    I_ = h**3 / 12
    adim_stiffs = []
    for i, L in enumerate(Ls):
        adim_stiff = Ks[:, i] * L**3 / (3 * E * I_) / 3
        adim_stiffs.append(adim_stiff)
    do_stiffness_variation_plotting(h_l_ratios, adim_stiffs, L_names)


if __name__ == "__main__":
    main()
