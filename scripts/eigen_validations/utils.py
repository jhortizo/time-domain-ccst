import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import plot_fields_quad9_rot4


def check_eigenvals_convergence(
    geometry_type,
    params,
    cst_model,
    constraints_loads,
    materials,
    eigsolution,
    force_reprocess,
    mesh_sizes,
    plot_style="none",
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
            eigensolution=eigsolution,
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

    first_eigvals = [eigvals[0] for eigvals in eigvalss]
    first_eigvals_diff = np.diff(first_eigvals)
    first_eigvals_diff = np.insert(first_eigvals_diff, 0, np.nan)

    second_eigvals = [eigvals[1] for eigvals in eigvalss]
    second_eigvals_diff = np.diff(second_eigvals)
    second_eigvals_diff = np.insert(second_eigvals_diff, 0, np.nan)

    tenth_eigvals = [eigvals[9] for eigvals in eigvalss]
    tenth_eigvals_diff = np.diff(tenth_eigvals)
    tenth_eigvals_diff = np.insert(tenth_eigvals_diff, 0, np.nan)

    df = pd.DataFrame(
        {
            "Mesh Size": mesh_sizes,
            "Number of Elements": n_elements,
            "First Eigenvalue": first_eigvals,
            "First Eigenvalue Diff": first_eigvals_diff,
            "Second Eigenvalue": second_eigvals,
            "Second Eigenvalue Diff": second_eigvals_diff,
            "Tenth Eigenvalue": tenth_eigvals,
            "Tenth Eigenvalue Diff": tenth_eigvals_diff,
        }
    )

    print(df)

    plt.figure()
    plt.plot(n_elements, tenth_eigvals, label="Tenth Eigenvalue")
    plt.xlabel("Mesh Size")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.grid()
    plt.show()
