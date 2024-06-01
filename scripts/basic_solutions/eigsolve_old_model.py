import warnings

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import plot_fields

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def main():
    geometry_type = "square"
    params = {"side": 1.0, "mesh_size": 0.1}
    force_reprocess = True
    cst_model = "cst_quad9"
    constraints_loads = "borders_fixed"
    eigsolution = True

    bc_array, eigvals, eigvecs, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        eigensolution=eigsolution,
        force_reprocess=force_reprocess,
    )

    n_eigvec = 0
    plot_fields(bc_array, nodes, elements, eigvecs[:, n_eigvec])


if __name__ == "__main__":
    main()
