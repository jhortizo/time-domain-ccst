import warnings

from time_domain_ccst.fem_solver import retrieve_solution
from time_domain_ccst.plotter import plot_fields_quad9_rot4

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def main():
    geometry_type = "rectangle"
    params = {"side_y": 1.0, "side_x": 3.0, "mesh_size": 0.5}
    force_reprocess = True
    cst_model = "cst_quad9_rot4"
    constraints_loads = "cantilever_support_load"

    bc_array, solution, nodes, elements = retrieve_solution(
        geometry_type,
        params,
        cst_model,
        constraints_loads,
        force_reprocess=force_reprocess,
    )

    plot_fields_quad9_rot4(bc_array, nodes, elements, solution)


if __name__ == "__main__":
    main()
