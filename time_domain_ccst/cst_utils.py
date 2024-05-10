import numpy as np
from solidspy import gaussutil as gau
from solidspy.femutil import jacoper, shape_quad9, shape_quad4


def custom_eqcounter(cons, vertex_nodes_ids, ndof_node=2):
    """Count active equations

    Creates boundary conditions array bc_array

    This custom version is used specifically for the CST element
    where the rotation dof is only present in the vertex nodes.

    Parameters
    ----------
    cons : ndarray.
      Array with constraints for each node.

    Returns
    -------
    neq : int
      Number of equations in the system after removing the nodes
      with imposed displacements.
    bc_array : ndarray (int)
      Array that maps the nodes with number of equations.

    """
    nnodes = cons.shape[0]
    bc_array = cons.copy().astype(int)
    # mark all rotations from nodes not in vertex_nodes_ids as constrained
    edge_nodes_ids = np.setdiff1d(range(nnodes), vertex_nodes_ids)
    bc_array[edge_nodes_ids, 2] = -1

    neq = 0
    for i in range(nnodes):
        for j in range(ndof_node):
            if bc_array[i, j] == 0:
                bc_array[i, j] = neq
                neq += 1

    return neq, bc_array


def assem_op_cst_quad9_rot4(cons, elements):
    """Create assembly array operator

    Count active equations, create boundary conditions array ``bc_array``
    and the assembly operator DME.

    Parameters
    ----------
    cons : ndarray.
      Array with constraints for each degree of freedom in each node.
    elements : ndarray
      Array with the number for the nodes in each element.

    Returns
    -------
    assem_op : ndarray (int)
      Assembly operator.
    bc_array : ndarray (int)
      Boundary conditions array.
    neq : int
      Number of active equations in the system.

    """
    nels = elements.shape[0]
    vertex_nodes_ids = list(set(elements[:, 3:7].flatten()))
    assem_op = np.zeros([nels, 23], dtype=np.integer)
    neq, bc_array = custom_eqcounter(cons, vertex_nodes_ids, ndof_node=3)
    for ele in range(nels):
        assem_op[ele, :22] = bc_array[elements[ele, 3:]].flatten()
        assem_op[ele, 27] = neq + ele
    return assem_op, bc_array, neq + nels


def cst_mat_2d_u_w(r, s, coord, element_u, element_w):
    """
    Interpolation matrices for a quadratic element for plane
    c-cst elasticity

    Considers the use of an element for
    the displacement field and different one
    for the rotation field.

    Parameters
    ----------
    r : float
        Horizontal coordinate of the evaluation point.
    s : float
        Vertical coordinate of the evaluation point.
    coord : ndarray (float)
        Coordinates of the element.
    element_u : callable
        Element for the displacement field.
    element_w : callable
        Element for the rotation field.

    Returns
    -------
    H : ndarray (float)
        Array with the shape functions evaluated at the point (r, s)
        for each degree of freedom.
    B : ndarray (float)
        Array with the displacement to strain matrix evaluated
        at the point (r, s).
    det : float
        Determinant of the Jacobian.

    References
    ----------

    .. [CST] Ali R. Hadhesfandiari, Gary F. Dargush.
        Couple stress theory for solids. International Journal
        for Solids and Structures, 2011, 48, 2496-2510.
    """
    N_u, dN_udr = element_u(r, s)
    N_w, dN_wdr = element_w(r, s)

    det_u, jaco_inv_u = jacoper(dN_udr, coord)
    det_w, jaco_inv_w = jacoper(dN_wdr, coord)

    dN_udr = jaco_inv_u @ dN_udr
    dN_wdr = jaco_inv_w @ dN_wdr

    H = np.zeros((2, 2 * N_u.shape[0]))
    B = np.zeros((3, 2 * N_u.shape[0]))
    Bk = np.zeros((2, N_w.shape[0]))
    B_curl = np.zeros((2 * N_u.shape[0],))

    H[0, 0::2] = N_u
    H[1, 1::2] = N_u

    B[0, 0::2] = dN_udr[0, :]
    B[1, 1::2] = dN_udr[1, :]
    B[2, 0::2] = dN_udr[1, :]
    B[2, 1::2] = dN_udr[0, :]

    Bk[0, :] = -dN_wdr[1, :]
    Bk[1, :] = dN_wdr[0, :]

    B_curl[0::2] = -dN_udr[1, :]
    B_curl[1::2] = dN_udr[0, :]
    return N_u, N_w, H, B, Bk, B_curl, det_u, det_w


def cst_quad9_rot4(coord, params):
    """
    Quadrilateral element with 9 nodes for Corrected Couple-Stress
    elasticity (C-CST) under plane-strain as presented in [CST]_

    Displacement fields (x,y) are sampled in the nine nodes of the
    element, the rotation field is sampled in the element vertices only,
    skew symmetric part of force-stress tensor is sampled in the center node only.

    Parameters
    ----------
    coord : coord
        Coordinates of the element.
    params : list
        List with material parameters in the following order:
        [Young modulus, Poisson coefficient, couple modulus,
         bending modulus, mass density, inertia density].

    Returns
    -------
    stiff_mat : ndarray (float)
        Local stifness matrix.
    mass_mat : ndarray (float)
        Local mass matrix.

    References
    ----------
    .. [CST] Ali R. Hadhesfandiari, Gary F. Dargush.
        Couple stress theory for solids. International Journal
        for Solids and Structures, 2011, 48, 2496-2510.
    """
    E, nu, eta, rho = params
    stiff_mat = np.zeros((23, 23))  # 18 for u, 4 for w, 1 for s
    mass_mat = np.zeros((23, 23))
    c = (
        E
        * (1 - nu)
        / ((1 + nu) * (1 - 2 * nu))
        * np.array(
            [
                [1, nu / (1 - nu), 0],
                [nu / (1 - nu), 1, 0],
                [0, 0, (1 - 2 * nu) / (2 * (1 - nu))],
            ]
        )
    )
    b = 4 * eta * np.eye(2)
    npts = 3
    gpts, gwts = gau.gauss_nd(npts)
    for cont in range(0, npts**2):
        r = gpts[cont, 0]
        s = gpts[cont, 1]
        _, N_w, H, B, Bk, B_curl, det_u, det_w = cst_mat_2d_u_w(
            r, s, coord, shape_quad9, shape_quad4
        )
        Ku = B.T @ c @ B
        Kw = Bk.T @ b @ Bk
        K_w_s = -2 * N_w
        factor_u = det_u * gwts[cont]
        factor_w = det_w * gwts[cont]
        stiff_mat[0:18, 0:18] += factor_u * Ku
        stiff_mat[18:23, 18:23] += factor_w * Kw
        stiff_mat[0:18, 23] += factor_u * B_curl.T
        stiff_mat[18:23, 23] += factor_w * K_w_s.T
        stiff_mat[23, 0:18] += factor_u * B_curl
        stiff_mat[23, 18:23] += factor_w * K_w_s
        mass_mat[0:18, 0:18] += rho * factor_u * (H.T @ H)
    return stiff_mat, mass_mat
