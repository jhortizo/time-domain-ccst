import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time_domain_ccst.fem_solver import retrieve_solution


def main():
    geometry_type = "rectangle"
    force_reprocess = False
    cst_model = "cst_quad9_rot4"
    constraints_loads = "cantilever_support_load"
    eigsolution = True

    E = 1
    nu = 0
    rho = 1
    mu = E / (2 * (1 + nu))
    mesh_size = 0.65

    h = 1.0
    L = 10 * h
    params = {"side_y": h, "side_x": L, "mesh_size": mesh_size}

    h_l_ratios = np.logspace(4, -4, 17)
    ls = h / h_l_ratios
    etas = ls**2 * mu

    eigvalss = []
    for i in tqdm(range(len(h_l_ratios)), desc="h/l ratios"):
        materials = np.array([[E, nu, etas[i], rho]])
        custom_str = f"h_l_ratio_{h_l_ratios[i]:.2e}"

        bc_array, eigvals, eigvecs, nodes, elements = retrieve_solution(
            geometry_type,
            params,
            cst_model,
            constraints_loads,
            materials,
            eigensolution=eigsolution,
            force_reprocess=force_reprocess,
            custom_str=custom_str,
        )

        eigvalss.append(eigvals[:1000])

    eigvalss = np.array(eigvalss)

    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1000),  eigvalss[[0, -1], :].T, '-')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Value')
    plt.legend([f"h/l = {h_l_ratios[i]:.0e}" for i in [0, -1]]
                )
    plt.grid()
    plt.savefig('compare_cantilever_eigenvalues.png', dpi=300)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(h_l_ratios, eigvalss[:, 2:7], "o-")
    plt.xlabel("h/l")
    plt.ylabel("Eigenvalues")
    plt.legend([f"Eigenvalue {i}" for i in range(3, 8)])
    plt.xscale("log")
    plt.grid()
    plt.savefig('eigenvalue_vs_hlratio.png', dpi=300)
    plt.show()




if __name__ == "__main__":
    main()
