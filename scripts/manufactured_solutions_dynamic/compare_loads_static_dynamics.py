"""
This file exists only as a mean to verify that the loads generated in the mms static files are congruent with the ones
in dynamic cases. It is not important, can and should be omitted in final versions of this repo, or converted into a test 
or something like that...
"""

import numpy as np
from solidspy.assemutil import assembler, loadasem
from solidspy.postprocesor import complete_disp, plot_node_field

from time_domain_ccst.cst_utils import assem_op_cst_quad9_rot4, cst_quad9_rot4
from time_domain_ccst.mms.proposed_solutions import (
    manufactured_solution_no_oscillations as ms_s,
)
from time_domain_ccst.mms.utils import (
    calculate_body_force_fcn_continuum_mechanics as cb_s,
)
from time_domain_ccst.mms.utils import (
    generate_load_mesh as g_s,
)
from time_domain_ccst.mms.utils import (
    impose_body_force_loads as i_s,
)
from time_domain_ccst.mms_t.proposed_solutions import (
    manufactured_solution_no_oscillations as ms_t,
)
from time_domain_ccst.mms_t.utils import (
    calculate_body_force_fcn_continuum_mechanics as cb_t,
)
from time_domain_ccst.mms_t.utils import (
    generate_load_mesh as g_t,
)
from time_domain_ccst.mms_t.utils import (
    impose_body_force_loads as i_t,
)

mesh_size = 0.5
E = 1
nu = 0.3
rho = 1
eta = 1
omega = 1

mats = [
    E,
    nu,
    eta,
    rho,
]

mats = np.array([mats])

# static
u_s, u_fcn_s, _ = ms_s()
body_force_fcn_s, f_s = cb_s(u_s)
cons, elements, nodes, loads = g_s(mesh_size, "bla.msh")
loads_s = i_s(loads, nodes, body_force_fcn_s, elements)

assem_op, bc_array, neq = assem_op_cst_quad9_rot4(cons, elements)

stiff_mat, mass_mat = assembler(
    elements, mats, nodes, neq, assem_op, uel=cst_quad9_rot4
)
rhs = loadasem(loads_s, bc_array, neq)

rhs_s = mass_mat @ rhs

loads_field = complete_disp(bc_array, nodes, rhs_s, ndof_node=2)
plot_node_field(loads_field, nodes, elements, title=["loads_x", "loads_y "])


# dynamic
u_t, u_fcn_t = ms_t()
body_force_fcn_t, f_t = cb_t(u_t)

cons, elements, nodes, loads = g_t(mesh_size, "bla2.msh")
loads_t = i_t(loads, nodes, body_force_fcn_t, 0.0)

assem_op, bc_array, neq = assem_op_cst_quad9_rot4(cons, elements)

stiff_mat, mass_mat = assembler(
    elements, mats, nodes, neq, assem_op, uel=cst_quad9_rot4
)
rhs = loadasem(loads_t, bc_array, neq)
# approximation to eval the function weighted by the form functions
rhs_t = mass_mat @ rhs

from solidspy.postprocesor import complete_disp, plot_node_field

loads_field = complete_disp(bc_array, nodes, rhs_t, ndof_node=2)
plot_node_field(loads_field, nodes, elements, title=["loads_x", "loads_y "])


pass
