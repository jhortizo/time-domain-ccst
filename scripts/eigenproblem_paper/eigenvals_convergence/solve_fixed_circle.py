import numpy as np
import pandas as pd
from utils import calculate_save_eigvals, plot_convergence


def main():
    geometry_type = "circle"
    force_reprocess = False
    params = {"radius": 1.0}
    cst_model = "cst_quad9_rot4"
    constraints_loads = "circle_borders_fixed"

    data_folder = "data/solutions"
    filename = f"{data_folder}/{constraints_loads}.csv"

    materials = np.array(
        [
            [
                1,  # E, young's modulus
                0.29,  # nu, poisson's ratio
                1,  # eta, coupling parameter
                1,  # rho, density
            ]
        ]
    )

    mesh_sizes = np.logspace(0, -0.75, 9)[:-2]
    plot_style = "last"

    # if filename exists then load it, else generate it
    if not force_reprocess:
        try:
            df = pd.read_csv(filename)
            print("Loaded dataframe from", filename)
        except FileNotFoundError:
            print("File not found, generating it...")
            df = calculate_save_eigvals(
                geometry_type,
                params,
                cst_model,
                constraints_loads,
                materials,
                force_reprocess,
                mesh_sizes,
                plot_style,
                df_filename=filename,
            )
    else:
        df = calculate_save_eigvals(
            geometry_type,
            params,
            cst_model,
            constraints_loads,
            materials,
            force_reprocess,
            mesh_sizes,
            plot_style,
            df_filename=filename,
        )

    plot_convergence(df, eigval=10, custom_str=constraints_loads)



if __name__ == "__main__":
    main()
