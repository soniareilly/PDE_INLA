import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time
from hippylib import *
import hippylib_changes as hc
from hippylib.modeling.variables import PARAMETER
from hippylib.modeling.reducedHessian import ReducedHessian
from hippylib.algorithms.lowRankOperator import LowRankOperator
from hippylib.modeling.posterior import GaussianLRPosterior
from hippylib.algorithms.multivector import MultiVector
from hippylib.utils.random import parRandom
from experiment_config import AdvDiffSetup, LowRankObjective

def setup_adv_diff(config, *, velocity_plot=False):
    """
    Load in saved meshes and targets, load (3D) or compute (2D) wind velocity
    """
    dim = config.dim
    mesh_path = config.mesh_path
    target_path = config.target_path

    if dim == 2:
        ## Import 2D mesh
        mesh = dl.refine( dl.Mesh(mesh_path) )
        Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
        Vh2 = dl.FunctionSpace(mesh, "Lagrange", 2)

        ## Advection velocity field
        wind_velocity = hc.computeVelocityField(mesh)

        ## Initial Condition
        ic_expr = dl.Expression(
            'std::min(0.5,std::exp(-100*(std::pow(x[0]-0.35,2) +  std::pow(x[1]-0.7,2))))',
            element=Vh2.ufl_element())
    elif dim == 3:
        ## Import 3D mesh and advection velocity field (precomputed)
        mesh = dl.Mesh()
        hdf = dl.HDF5File(mesh.mpi_comm(), mesh_path, "r")
        hdf.read(mesh, "/mesh", False)
        Xh = dl.VectorFunctionSpace(mesh,'Lagrange', 2)
        Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
        Vh2 = dl.FunctionSpace(mesh, "Lagrange", 2)

        ## Advection velocity field
        wind_velocity = dl.Function(Xh)
        hdf.read(wind_velocity, "/velocity")
        hdf.close()

        ## Initial Condition
        center = ((0.15,0.85,0.7))
        width = 50.0
        cutoff = 0.5
        ic_expr = dl.Expression(
            "std::min(cutoff, std::exp(-a * (std::pow(x[0]-x0, 2) + std::pow(x[1]-y0, 2) + std::pow(x[2]-z0, 2))))",
            a=width, x0=center[0], y0=center[1], z0=center[2], cutoff = 0.5,
            element=Vh2.ufl_element()
        )
    else:
        raise ValueError("Dimension must be either 2 or 3")

    print("Number of elements: {0}".format(mesh.num_cells()))
    print("Number of dofs, first order: {0}".format(Vh.dim()))
    print("Number of dofs, second order: {0}".format(Vh2.dim()))

    true_initial_condition = dl.interpolate(ic_expr, Vh).vector()

    ## Observation points
    targets = np.loadtxt(target_path)

    if dim == 2 and velocity_plot:
        Xh = dl.VectorFunctionSpace(mesh,'Lagrange', 2)
        vh = dl.project(wind_velocity,Xh)
        dl.plot(vh)
        plt.scatter(targets[:,0],targets[:,1],color='red')

    return AdvDiffSetup(mesh=mesh, Vh=Vh, Vh2=Vh2, wind_velocity=wind_velocity, 
                        true_initial_condition=true_initial_condition, targets=targets)

def solve_fwd_problem(setup, config, *, plot=True, save=False):
    dim = config.dim; mesh_vertices = config.mesh_vertices
    sigma_true = config.sigma_true; prior_mean = config.prior_mean; kappa = config.kappa
    nt = config.time.nt; t_init = config.time.t_init; t_final = config.time.t_final
    first_observation_time = config.time.first_observation_time
    observation_dt = config.time.observation_dt
    mesh = setup.mesh; Vh = setup.Vh; Vh2 = setup.Vh2
    wind_velocity = setup.wind_velocity
    true_initial_condition = setup.true_initial_condition
    targets = setup.targets

    dt = t_final/nt
    simulation_times = np.arange(t_init, t_final+.5*dt, dt)
    observation_times = np.arange(first_observation_time, t_final+.5*dt, observation_dt)
    print(observation_times)

    print ("Number of observation points: {0}".format(targets.shape[0]) )
    print ("Number of observation times: {0}".format(observation_times.shape[0]) )
    # initialize observations
    misfit = hc.SpaceTimePointwiseStateObservation(Vh2, observation_times, targets)

    # Prior required by TimeDependentAD during setup (not used)
    # covariance C = ((delta * (I - eta * Laplacian))^{-2}
    prior_eta = 0.125
    prior_delta = 8.0
    prior = hc.BiLaplacianPrior(Vh, prior_eta*prior_delta, prior_delta, robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(prior_mean), Vh).vector()

    problem_true = hc.TimeDependentAD(mesh, [Vh2,Vh,Vh2], prior, misfit, simulation_times, kappa, wind_velocity, True)

    # initialize vector in the state space
    utrue = problem_true.generate_vector(STATE)
    x = [utrue, true_initial_condition, None]
    # solve forward problem
    problem_true.solveFwd(x[STATE], x)
    # observe solution and add error
    misfit.observe(x, misfit.d)
    parRandom.normal_perturb(sigma_true,misfit.d)
    misfit.noise_variance = sigma_true**2

    # Plot/save forward solution
    if plot:
        if dim == 2:
            ic_func = dl.Function(Vh)
            ic_func.vector()[:] = true_initial_condition
            ic_Vh2 = dl.project(ic_func, Vh2).vector()
            hc.show_solution(Vh2, ic_Vh2, utrue, mytitle="Solution")
        else:
            print('Plotting forward solution is not enabled in 3D.')
    if save:
        if dim == 2:
            print("Saving 2D forward solution not implemented.")
        if dim == 3:
            # Create the PVD file
            file_pvd = dl.File("forward_sol_{0}.pvd".format(mesh_vertices))
            # Iterate through the time steps stored in 'x'
            # x[STATE] is the TimeDependentVector object
            for i, t in enumerate(simulation_times):
                u_plot = dl.Function(Vh2)
                # access the .data list directly
                vec_at_t = x[STATE].data[i]
                # Copy values into the Function's vector
                u_plot.vector()[:] = vec_at_t
                u_plot.rename("concentration", "label")
                # sanity check that max concentration is decreasing
                print(f"Time {t}: Max concentration = {vec_at_t.norm('linf')}")
                # Write to PVD
                file_pvd << (u_plot, t)

    # Precompute -A^T y in MAP point
    # note -- problem contains the true noise variance in misfit. Will be overwritten in computation.
    problem = hc.TimeDependentAD(mesh, [Vh2,Vh,Vh2], prior, misfit, simulation_times, kappa, wind_velocity, True)
    problem.misfit.noise_variance = 1
    # This computes -A^T Q_eps y, so -A^T y with sigma = 1
    [u0,m0,neg_adj_y] = problem.generate_vector()
    problem.solveFwd(u0, [u0,m0,neg_adj_y])
    problem.solveAdj(neg_adj_y, [u0,m0,neg_adj_y]) 

    return problem, neg_adj_y

def ComputePosterior(theta, lmbda, V, neg_adj_y, pretheta, problem, use_CG=False, omega_full=None):
    '''
    Solve inverse problem
    Output: posterior object and mg = mu_pr^T Q_pr + y^T Q_eps A
    Input:  theta = eta, delta: hyperparameters of prior, sigma: noise hyperparameter
            lmbda, V: low rank decomp of Q_pre^-1/2 A^T A Q_pre^-1/2, where Q_pre is a preconditioning prior precision
            pretheta = preeta, predelta: parameters of Q_pre, and presigma: noise stdev used in low rank approx
            problem: contains mesh, Vstate, Vparam, misfit, simulation_times, kappa, wind_velocity, and a prior that is overwritten
    '''
    preeta, predelta, presigma = pretheta
    eta, delta, sigma = theta
    
    prior = hc.BiLaplacianPrior(problem.Vh[PARAMETER], eta*delta, delta, robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(0.), problem.Vh[PARAMETER]).vector()

    problem.prior = prior
    problem.misfit.noise_variance = sigma**2

    ## Compute the gradient
    # p = -A^T Q_eps y
    [u,m,p] = problem.generate_vector()
    p = neg_adj_y.copy()
    for i in range(p.nsteps):
        p.data[i] *= 1/sigma**2
    # mg = -Q_pr mu_pr - A^T Q_eps y
    mg = problem.generate_vector(PARAMETER)
    grad_norm = problem.evalGradientParameter([u,m,p], mg)

    ## Compute posterior precision
    # matrix free application of posterior precision/covariance
    H = ReducedHessian(problem, misfit_only=True) 
    
    # prior preconditioning
    if preeta == eta and predelta == delta:
        lmbda_new = lmbda
        V_new = V
    # weakest or no preconditioning
    else:
        preprior = hc.BiLaplacianPrior(problem.Vh[PARAMETER], preeta*predelta, predelta, robin_bc=True)
        preprior.mean = dl.interpolate(dl.Constant(0.), problem.Vh[PARAMETER]).vector()
        # replace preprior with prior
        W = MultiVector(V)
        for i in range(V.nvec()):
            preprior.R.mult(V[i], W[i])
        H_temp = LowRankOperator(lmbda, W)
        k = V.nvec()
        pad = 20 
        if omega_full is None:
            raise ValueError("omega_full is required for fixed-basis preconditioning.")
        elif k + pad > omega_full.nvec():
            raise ValueError("omega_full does not contain enough vectors.")
        Omega = multivector_slice(omega_full, k + pad)
        lmbda_new, V_new = hc.singlePassG(H_temp, prior.R, prior.Rsolver, Omega, k)
    # correcting for noise stdev used in low rank approx (presigma)
    lmbda_new = lmbda_new*(presigma**2)/(sigma**2)
    posterior = GaussianLRPosterior(prior, lmbda_new, V_new)

    # Compute posterior mean
    if use_CG:
        H.misfit_only = False
        solver = CGSolverSteihaug()
        solver.set_operator(H)
        solver.set_preconditioner( posterior.Hlr )
        solver.parameters["print_level"] = -1
        solver.parameters["rel_tolerance"] = 1e-6
        solver.solve(m, -mg)
    else:
        H = posterior.Hlr
        H.solve(m,-mg)
    posterior.mean = m
    
    return posterior,mg,lmbda_new,V_new

# Helper function for slicing multivectors up to vector n, inclusive
def multivector_slice(mv, n):
    mv_n = MultiVector(mv[0], n)
    for i in range(n):
        mv_n[i].zero()
        mv_n[i].axpy(1.0, mv[i])
    return mv_n

def rank_from_cutoff(eigenvalues, cutoff):
    """Return the first rank at which the spectrum falls below cutoff."""
    below_cutoff = np.flatnonzero(eigenvalues < cutoff)
    if len(below_cutoff) == 0:
        print("Approximation is too low rank, cutoff eigval not achieved")
        return len(eigenvalues)-1
    return int(below_cutoff[0])

def calibrate_rank(pretheta, cutoff, calibration_rank, problem):
    lmbda, V = LowRankApprox(pretheta, calibration_rank, problem)
    rank = rank_from_cutoff(lmbda, cutoff)
    lmbda = lmbda[0:rank]
    V = multivector_slice(V,rank)
    return lmbda, V

def calibrate_rank_PP(thetas, cutoff, calibration_rank, problem):
    ranks = [0 for idx in thetas]
    for idx in range(len(thetas)):
        theta = thetas[idx]
        lmbda, V = LowRankApprox(theta, calibration_rank, problem)
        ranks[idx] = rank_from_cutoff(lmbda, cutoff)
    return max(ranks)

# -log pi(theta), in this case independent uniform distributions on eta, delta, sigma
def neg_log_hyperprior(theta, hyp_pr_params):
    min_eta = hyp_pr_params["min_eta"]; max_eta = hyp_pr_params["max_eta"]
    min_del = hyp_pr_params["min_delta"]; max_del = hyp_pr_params["max_delta"]
    min_sig = hyp_pr_params["min_sigma"]; max_sig = hyp_pr_params["max_sigma"]
    eta, delta, sigma = theta
    theta_prior = np.log(max_eta-min_eta) + np.log(max_del-min_del) + np.log(max_sig-min_sig)
    # set prior value to 0 outside the domain of the prior (neg log value to large)
    if eta < min_eta or eta > max_eta or delta < min_del or delta > max_del or sigma < min_sig or sigma > max_sig:
        theta_prior = 1e30
    return theta_prior

# -log pi(theta | y) (- log posterior marginal joint pdf of theta)
# warning: changes noise variance in problem.misfit for each theta
def neglogpi_theta(theta, lmbda, V, tol, neg_adj_y, pretheta, hyp_pr_params, problem, use_CG=False, omega_full = None):
    preeta, predelta, presigma = pretheta
    eta, delta, sigma = theta

    # make copy of lmbda, V truncated to new theta
    cutoff = tol*sigma**2*delta**2/presigma**2/predelta**2
    rank = rank_from_cutoff(lmbda, cutoff)
    lmbda_new = lmbda[0:rank]
    V_new = multivector_slice(V,rank)

    # compute new posterior
    posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda_new, V_new, neg_adj_y, pretheta, problem, use_CG, omega_full=omega_full)
    
    # -log(|Q_pr|/|Q_post|)
    det_ratio = 0.0
    for ll in posterior.d:
        det_ratio += np.log(1+ll)
    # -log|Q_eps| = 2*n_obs*log(sigma)
    det_ratio += 2*problem.misfit.ntargets.shape[0]*problem.misfit.observation_times.shape[0]*np.log(sigma)
    det_ratio *= 0.5

    # -log pdf of theta prior
    theta_prior = neg_log_hyperprior(theta, hyp_pr_params)

    # -mu_post^T Q_post mu_post 
    uQu = 0.5*mg.inner(posterior.mean)

    # mu_pr^T Q_pr mu_pr
    Qmu = dl.Vector(posterior.prior.R.mpi_comm())
    posterior.prior.init_vector(Qmu,0)
    posterior.prior.R.mult(posterior.prior.mean,Qmu)
    muQmu = 0.5*posterior.prior.mean.inner(Qmu)

    # y^T Q_eps y
    yQy = 0.5*problem.misfit.d.inner(problem.misfit.d)/(sigma**2)

    return det_ratio + theta_prior + uQu + muQmu + yQy

# find rank k approx to Hessian preconditioned by prior with params pretheta
# eigvals are lambda/presigma^2/predelta^2, using defn of lambda from paper
def LowRankApprox(pretheta, k, problem):
    preeta, predelta, presigma = pretheta

    preprior = hc.BiLaplacianPrior(problem.Vh[PARAMETER], preeta*predelta, predelta, robin_bc=True)
    preprior.mean = dl.interpolate(dl.Constant(0.), problem.Vh[PARAMETER]).vector()
    problem.prior = preprior
    problem.misfit.noise_variance = presigma**2
        
    H_misfit_only = ReducedHessian(problem, misfit_only=True)
    pad = int(k/2)
    Omega = MultiVector(problem.generate_vector(PARAMETER), k+pad)
    parRandom.normal(1., Omega)

    lmbda, V = hc.singlePassG(H_misfit_only, preprior.R, preprior.Rsolver, Omega, k) 
    return lmbda, V

# returns errors in posterior covariance for ranks ks, first rank at which error is below threshold,
# and low rank approximation with rank max(ks)
def PostCovError(theta, lmbda, V, neg_adj_y, pretheta, truth, ks, threshold, problem):
    errs = np.zeros(len(ks))
    first_ii = -1
    for ii in range(len(ks)):
        print(ii)
        k = ks[ii]
        posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda[0:k], multivector_slice(V,k), neg_adj_y, pretheta, problem)
        posterior_trace,pr_tr,corr_tr = posterior.trace(method="Exact")
        errs[ii] = (posterior_trace-truth)/truth
        # first rank with error below threshold
        if first_ii < 0 and errs[ii] < threshold:
            first_ii = ii
    if first_ii > 0:
        min_k = ks[first_ii]
    else:
        min_k = None
        print(f'did not reach threshold before rank {max(ks)}')
    return errs, min_k

def make_neg_log_pi_theta(preconditioner, rank_mode, calibration_rank, calibrated_ranks, thetas, low_rank_tolerance, neg_adj_y, hyp_pr_params, problem, use_CG = False):
    """
    Make a single-input function for -log pi(theta)
    """
    if preconditioner == "prior":
        pretheta = None
    elif preconditioner == "weakest":
        pretheta = np.array([hyp_pr_params["min_eta"], 1.0, 1.0])
    elif preconditioner == "unpreconditioned":
        pretheta = np.array([0.0, 1.0, 1.0])
    else:
        raise ValueError("preconditioner must be 'prior', 'weakest', or 'unpreconditioned'")

    # precompute low-rank approx for WP or UP
    if preconditioner == 'weakest' or preconditioner == 'unpreconditioned':
        cutoff = low_rank_tolerance * hyp_pr_params["min_sigma"]**2 * hyp_pr_params["min_delta"]**2
        if rank_mode == 'calibrate':
            lmbda, V = calibrate_rank(pretheta, cutoff, calibration_rank, problem)
            print(f"Computing low rank approximation to rank {len(lmbda)}")
        elif rank_mode == "fixed":
            rank = calibrated_ranks[preconditioner]
            lmbda, V = LowRankApprox(pretheta, rank, problem)
        else:
            raise ValueError("rank_mode must be calibrate or fixed")
        rank_upper_bound = len(lmbda)       # not used for WP, UP, but still a required input
        pad = 20
        omega_full = MultiVector(problem.generate_vector(PARAMETER), len(lmbda) + pad)
        parRandom.normal(1.0, omega_full)

    # PP handled separately, no precomputation of low rank approx
    if preconditioner == 'prior':
        cutoff = low_rank_tolerance
        if rank_mode == 'calibrate':
            rank_upper_bound = calibrate_rank_PP(thetas, cutoff, calibration_rank, problem)
            print(f"Computing low rank approximations to rank {rank_upper_bound}")
        elif rank_mode == 'fixed':
            rank_upper_bound = calibrated_ranks['prior']
        else:
            raise ValueError("rank_mode must be calibrate or fixed")
        lmbda = None; V = None
        omega_full = None

    def neglogpi_helper(theta):
        if preconditioner == 'prior':
            current_lmbda, current_V = LowRankApprox(theta.tolist(), rank_upper_bound, problem)
        else:
            current_lmbda = lmbda; current_V = V
        return neglogpi_theta(theta, current_lmbda, current_V, low_rank_tolerance, neg_adj_y, pretheta, hyp_pr_params, problem, use_CG, omega_full)

    return LowRankObjective(objective=neglogpi_helper, eigenvalues=lmbda, eigenvectors=V, 
                            pretheta=pretheta, sketching_matrix=omega_full)


def evaluate_pi_theta_on_grid(bounds, num_values, neglogpi_helper,*, plot=True, save=False):
    eta_range = np.linspace(bounds["eta"][0], bounds["eta"][1], num_values["eta"])
    delta_range = np.linspace(bounds["delta"][0], bounds["delta"][1], num_values["delta"])
    sigma_range = np.linspace(bounds["sigma"][0], bounds["sigma"][1], num_values["sigma"])
    logpi = np.zeros((num_values["eta"],num_values["delta"],num_values["sigma"]))
    print('Progress in indices computed from (0,0,0) to ({0},{1},{2}):'.format(num_values["eta"]-1,num_values["delta"]-1,num_values["sigma"]-1))
    for i in range(len(eta_range)):
        for j in range(len(delta_range)):
            for k in range(len(sigma_range)):
                theta = np.array([eta_range[i],delta_range[j],sigma_range[k]])
                logpi[i,j,k] = neglogpi_helper(theta)
                print('({0},{1},{2})'.format(i,j,k))

    # scaled to have max value 1 in order to avoid overflow errors
    pitheta = np.exp(-logpi+np.min(logpi))

    if plot:
        # plot pi_theta as a function of eta and delta
        if num_values["eta"] > 2 and num_values["delta"] > 2:
            sig_idx = int(num_values["sigma"]/2)
            fig = plt.figure(figsize=(10,7.2))
            plt.rcParams.update({'font.size': 16})
            plt.set_cmap('bone')
            plt.pcolormesh(delta_range,eta_range,pitheta[:,:,sig_idx])
            plt.colorbar()
            plt.title(r'$\pi(\eta, \delta, \sigma | y)$')
            plt.ylabel(r'$\eta$')
            plt.xlabel(r'$\delta$')
        else:
            print("Too few eta or delta evaluation values for plotting.")

        # plot pi_theta as a function of sigma
        if num_values["sigma"] > 2:
            eta_idx = int(num_values["eta"]/2); d_idx = int(num_values["delta"]/2)
            fig = plt.figure(figsize=(10,7.2))
            plt.rcParams.update({'font.size': 16})
            plt.plot(sigma_range,pitheta[eta_idx,d_idx,:])
            plt.xlabel(r'$\sigma$')
            plt.ylabel(r'$\pi(\eta, \delta, \sigma|y)$')
        else:
            print("Too few sigma evaluation values for plotting.")

    if save:
        etamesh,dmesh,smesh = np.meshgrid(eta_range, delta_range, sigma_range, indexing='ij')
        header = "eta \t\t delta \t\t sigma \t\t pi_theta"
        filename = f"images/pi_theta_{num_values['eta']}x{num_values['delta']}x{num_values['sigma']}.txt"
        data = np.column_stack((etamesh.ravel(), dmesh.ravel(), smesh.ravel(), pitheta.ravel()))
        np.savetxt(filename, data, delimiter="\t", header=header, fmt=('%g', '%g', '%g', '%e'), comments="")

    return pitheta

def argmax_theta(neglogpi_helper, theta_init, optimization_method, optim_options, verbose=True):
    """
    Find the theta that maximizes pi(theta)
    """
    def opt_callback(intermediate_result):
        print(f"Current x: {intermediate_result.x}")
        print(f"Objective value: {intermediate_result.fun}")
    if verbose:
        callback = opt_callback
        optim_options['disp'] = True
    else:
        callback = None
        optim_options['disp'] = False

    opt_start = time.time()
    theta_opt = opt.minimize(neglogpi_helper,theta_init,method=optimization_method,callback=callback, options=optim_options)
    opt_end = time.time()

    if verbose:
        print(f"Optimization time: {opt_end-opt_start} seconds")
        print(f"MAP point of pi(theta|y): {theta_opt.x}")
    return theta_opt.x