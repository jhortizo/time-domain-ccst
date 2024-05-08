import numpy as np
from solidspy import gaussutil as gau
from solidspy.assemutil import eqcounter
from solidspy.femutil import jacoper, shape_quad9, shape_quad4


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
    assem_op = np.zeros([nels, 28], dtype=np.integer)
    neq, bc_array = eqcounter(cons, ndof_node=3)
    for ele in range(nels):
        assem_op[ele, :27] = bc_array[elements[ele, 3:]].flatten()
        assem_op[ele, 27] = neq + ele
    return assem_op, bc_array, neq + nels


def cst_mat_2d_u_t(r, s, coord, element_u):
    """
    Interpolation matrices for a quadratic element for plane
    c-cst elasticity

    Parameters
    ----------
    r : float
        Horizontal coordinate of the evaluation point.
    s : float
        Vertical coordinate of the evaluation point.
    coord : ndarray (float)
        Coordinates of the element.

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
    det_u, jaco_inv_u = jacoper(dN_udr, coord)
    dN_udr = jaco_inv_u @ dN_udr
    H = np.zeros((2, 2 * N_u.shape[0]))
    B = np.zeros((3, 2 * N_u.shape[0]))
    Bk = np.zeros((2, N_u.shape[0]))
    B_curl = np.zeros((2 * N_u.shape[0],))
    H[0, 0::2] = N_u
    H[1, 1::2] = N_u
    B[0, 0::2] = dN_udr[0, :]
    B[1, 1::2] = dN_udr[1, :]
    B[2, 0::2] = dN_udr[1, :]
    B[2, 1::2] = dN_udr[0, :]
    Bk[0, :] = -dN_udr[1, :]
    Bk[1, :] = dN_udr[0, :]
    B_curl[0::2] = -dN_udr[1, :]
    B_curl[1::2] = dN_udr[0, :]
    return N_u, H, B, Bk, B_curl, det_u


def cst_quad9_rot4(coord, params):
    """
    Quadrilateral element with 9 nodes for Corrected Couple-Stress
    elasticity (C-CST) under plane-strain as presented in [CST]_


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
    stiff_mat = np.zeros((28, 28))
    mass_mat = np.zeros((28, 28))
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
        N, H, B, Bk, B_curl, det = cst_mat_2d_u_t(r, s, coord, shape_quad9)
        Ku = B.T @ c @ B
        Kw = Bk.T @ b @ Bk
        K_w_s = -2 * N
        factor = det * gwts[cont]
        stiff_mat[0:18, 0:18] += factor * Ku
        stiff_mat[18:27, 18:27] += factor * Kw
        stiff_mat[0:18, 27] += factor * B_curl.T
        stiff_mat[18:27, 27] += factor * K_w_s.T
        stiff_mat[27, 0:18] += factor * B_curl
        stiff_mat[27, 18:27] += factor * K_w_s
        mass_mat[0:18, 0:18] += rho * factor * (H.T @ H)
    order = [
        0,
        1,
        18,
        2,
        3,
        19,
        4,
        5,
        20,
        6,
        7,
        21,
        8,
        9,
        22,
        10,
        11,
        23,
        12,
        13,
        24,
        14,
        15,
        25,
        16,
        17,
        26,
        27,
    ]
    stiff_mat = stiff_mat[:, order]
    mass_mat = mass_mat[:, order]
    return stiff_mat[order, :], mass_mat[order, :]
