import numpy as np

def hyper_marginal_Laplace_approx(neglogpi_theta, theta_MAP, dtheta):
    '''
    Returns eigendecomposition of inverse Hessian of -log pi(theta) at theta_MAP
    Inputs: neglogpi_theta -- negative log hyperparameter marginal. Should take only theta as input.
            theta_MAP -- argmax of neglogpi_theta
            dtheta -- vector of finite difference deltas used to approximate Hessian
    '''
    ntheta = np.size(dtheta)
    Hess_MAP = np.zeros((ntheta,ntheta))
    # compute necessary function evaluations for Hessian finite difference estimation
    neglogpiMAP = neglogpi_theta(theta_MAP)
    plustwo = np.zeros((ntheta,ntheta))
    plusone = np.zeros(ntheta)
    minusone = np.zeros(ntheta)
    for i in range(ntheta):
        dtheta_i = np.zeros(ntheta)
        dtheta_i[i] = dtheta[i]
        plusone[i] = neglogpi_theta(theta_MAP + dtheta_i)
        minusone[i] = neglogpi_theta(theta_MAP - dtheta_i)
        for j in range(i+1,ntheta):
            dtheta_i_j = np.zeros(ntheta)
            dtheta_i_j[i] = dtheta[i]; dtheta_i_j[j] = dtheta[j]
            plustwo[i,j] = neglogpi_theta(theta_MAP + dtheta_i_j)
            plustwo[j,i] = plustwo[i,j]
    # compute Hessian using precomputed function evaluations
    for i in range(ntheta):
        Hess_MAP[i,i] = (minusone[i] - 2*neglogpiMAP + plusone[i])/dtheta[i]**2
        for j in range(i+1,ntheta):
            Hess_MAP[i,j] = (plustwo[i,j] + neglogpiMAP - plusone[i] - plusone[j])/dtheta[i]/dtheta[j]
            Hess_MAP[j,i] = Hess_MAP[i,j]
    H_MAP_inv = np.linalg.inv(Hess_MAP)
    # find principal directions
    Hinv_lam,Hinv_V = np.linalg.eig(H_MAP_inv)
    return Hinv_lam, Hinv_V

def pt_pairs(list1, list2):
    '''
    Lists pairs of points given two lists of point locations
    e.g., inputs [[0 0],[1 1]] and [2 3], output [[0 0 2],[1 1 2],[0 0 3],[1 1 3]]
    first input must be list of lists, second must be list
    '''
    newlist = []
    for i in range(len(list1)):
        for j in range(len(list2)):
            newlist.append(list1[i]+[list2[j]])
    return newlist

def uniform_hyperprior_support(theta, hyp_pr_params):
    '''
    Checks whether theta is in the support of the uniform hyperprior.
    Used to avoid issues with quadrature points outside the support.
    '''
    min_eta, max_eta, min_del, max_del, min_sig, max_sig = hyp_pr_params
    if (theta[0] > min_eta and theta[1] > min_del and theta[2] > min_sig and 
        theta[0] < max_eta and theta[1] < max_del and theta[2] < max_sig):
        is_valid_point = True
    else:
        is_valid_point = False
    return is_valid_point

def find_quad_points(neglogpi_theta, theta_MAP, dtheta, delta_z, delta_pi, maxiter, in_bounds=None, scale=True):
    '''

    '''
    ntheta = np.size(dtheta)
    neglogpiMAP = neglogpi_theta(theta_MAP)

    Hinv_lam, Hinv_V = hyper_marginal_Laplace_approx(neglogpi_theta, theta_MAP, dtheta)
    Hinv_L_sqrt = np.diag(np.sqrt(Hinv_lam))

    def theta_of_z(z):
        return theta_MAP + np.dot(Hinv_V,np.dot(Hinv_L_sqrt,z))

    # for each coordinate of z, find its values with significant probability
    z_highprob = [np.array([0.0]) for i in range(ntheta)]
    for idx in range(ntheta):
        z = np.zeros(ntheta)
        z[idx] = delta_z
        count = 0
        while all(theta_of_z(z)>0) and neglogpi_theta(theta_of_z(z)) - neglogpiMAP < delta_pi and count < maxiter:
            z_highprob[idx] = np.append(z_highprob[idx],z[idx])
            z[idx] += delta_z
            count += 1
            print(count)
        count = 0
        z[idx] = -delta_z
        while all(theta_of_z(z)>0) and neglogpi_theta(theta_of_z(z)) - neglogpiMAP < delta_pi and count < maxiter:
            z_highprob[idx] = np.append(z_highprob[idx],z[idx])
            z[idx] -= delta_z
            count += 1
            print(count)

    # use to recursively find all combinations of possible points
    all_points = [[zval] for zval in z_highprob[0]]
    if ntheta > 1:
        for idx in range(1,ntheta):
            all_points = pt_pairs(all_points, z_highprob[idx])

    # search through them for only the ones with high enough probability
    # could be made more efficient -- don't recalculate along axes, and/or store values for later
    quad_points = []
    neglogpi_theta_quad = []
    print('Points to be checked: {0}'.format(len(all_points)))
    for i in range(len(all_points)):
        print(i)
        theta_i = theta_of_z(np.array(all_points[i]))
        if in_bounds is None:
            is_valid_point = True
        else:
            is_valid_point = in_bounds(theta_i)
        if is_valid_point:
            theta_i_func_val = neglogpi_theta(theta_i)
            if theta_i_func_val - neglogpiMAP < delta_pi:
                quad_points.append(theta_i)
                neglogpi_theta_quad.append(theta_i_func_val)
        else:
            print("found invalid point")
    quad_points = np.array(quad_points)
    neglogpi_theta_quad = np.array(neglogpi_theta_quad)
    pi_theta_quad = np.zeros(quad_points.shape[0])
    for i in range(quad_points.shape[0]):
        pi_theta_quad[i] = np.exp(-neglogpi_theta_quad[i]+neglogpiMAP)

    # optionally scale to integrate to 1 using these quadrature points
    d_area = np.sqrt(np.prod(Hinv_lam))
    Z = np.sum(pi_theta_quad)*d_area
    # scale evaluations of pi(theta|y)
    if scale:
        pi_theta_quad = pi_theta_quad/Z

    return quad_points, pi_theta_quad, d_area