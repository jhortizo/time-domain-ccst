import warnings

from time_domain_ccst.fem_solver import retrieve_solution

warnings.filterwarnings(
    "ignore", "The following kwargs were not used by contour: 'shading'", UserWarning
)  # ignore unimportant warning from solidspy


def main():
    geometry_type = "square"
    params = {"side": 1.0, "mesh_size": 1.0}
    force_reprocess = False

    bc_array, solution, nodes, elements = retrieve_solution(
        geometry_type, params, force_reprocess=force_reprocess
    )


if __name__ == "__main__":
    main()
