import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import plot_fields_quad9_rot4


def calculate_save_eigvals(
    geometry_type,
    params,
    cst_model,
    constraints_loads,
    materials,
    force_reprocess,
    mesh_sizes,
    plot_style="none",
    df_filename="eigvals_convergence.csv",
):


    eigvalss = []
    n_elements = []
    for mesh_size in tqdm(mesh_sizes, desc="Mesh sizes"):
        params["mesh_size"] = mesh_size
        print("Mesh size:", mesh_size)

        bc_array, eigvals, eigvecs, nodes, elements = retrieve_solution(
            geometry_type,
            params,
            cst_model,
            constraints_loads,
            materials,
            scenario_to_solve="eigenproblem",
            force_reprocess=force_reprocess,
        )
        print("Number of elements", len(elements))

        eigvalss.append(eigvals)
        n_elements.append(len(elements))

        if plot_style == "all":
            n_eigvec = 3
            plot_fields_quad9_rot4(
                bc_array, nodes, elements, eigvecs[:, n_eigvec], instant_show=False
            )
        elif plot_style == "last" and mesh_size == mesh_sizes[-1]:
            n_eigvec = 10
            plot_fields_quad9_rot4(
                bc_array, nodes, elements, eigvecs[:, n_eigvec], instant_show=False
            )

    eigvals_dict = {}

    for i in range(50):
        eigvals_i = [eigvals[i] for eigvals in eigvalss]
        eigvals_dict[f"eigval_{i+1}"] = eigvals_i

    df = pd.DataFrame(
        {
            "mesh_size": mesh_sizes,
            "n_elements": n_elements,
            **eigvals_dict,
        }
    )

    df.to_csv(df_filename, index=False)

    return df


def plot_convergence(
    df,
    eigval=10,
    custom_str="",
):
    plt.style.use("cst_paper.mplstyle")
    n_elements = df["n_elements"]
    eigvals = df["eigval_10"]

    plt.figure()
    plt.plot(n_elements, eigvals, label=f"Eigenvalues {eigval}")
    plt.xlabel("Number of Elements")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data/images/eigvals_convergence_{custom_str}.pdf", dpi=300)

