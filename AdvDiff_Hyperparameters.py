 # %%
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.optimize as opt
from scipy.integrate import trapezoid

import sys
import os
os.environ["HIPPYLIB_BASE_DIR"] = '/home/sonia/research/hyperparam_marginal/hippylib'
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR') )
# sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR') + "/applications/ad_diff/" )
from hippylib import *
# from model_ad_diff import SpaceTimePointwiseStateObservation, TimeDependentAD

from hippylib_changes import *

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
dl.set_log_active(False)

import time
np.random.seed(42)

# %%
# Helper function for slicing multivectors
def mv_k(mv, n):
    mv_n = MultiVector(mv[0], n)
    for i in range(n):
        mv_n[i].zero()
        mv_n[i].axpy(1.0, mv[i])
    return mv_n

def ComputePosterior(theta, lmbda, V, neg_adj_y, pretheta, problem, use_CG=False):
    '''
    Solve inverse problem
    Output: posterior object and mg = mu_u0^T Q_u0 + y^T Q_eps A
    Input:  theta = eta, delta: hyperparameters of prior, sigma: noise hyperparameter
            lmbda, V: low rank decomp of Q_pre^-1/2 A^T A Q_pre^-1/2, where Q_pre is a preconditioning prior precision
            pretheta = preeta, predelta: parameters of Q_pre, and presigma: noise stdev used in low rank approx
            problem: contains mesh, Vstate, Vparam, misfit, simulation_times, kappa, wind_velocity, and a prior that is overwritten
    '''
    preeta, predelta, presigma = pretheta
    eta, delta, sigma = theta
    
    prior = BiLaplacianPrior(problem.Vh[PARAMETER], eta*delta, delta, robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(prior_mean), problem.Vh[PARAMETER]).vector()

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
        preprior = BiLaplacianPrior(problem.Vh[PARAMETER], preeta*predelta, predelta, robin_bc=True)
        preprior.mean = dl.interpolate(dl.Constant(prior_mean), problem.Vh[PARAMETER]).vector()
        # replace preprior with prior
        W = MultiVector(V)
        for i in range(V.nvec()):
            preprior.R.mult(V[i], W[i])
        H_temp = LowRankOperator(lmbda, W)
        k = V.nvec()
        pad = 20 
        Omega = MultiVector(x[PARAMETER], k+pad)
        parRandom.normal(1., Omega)
        lmbda_new, V_new = singlePassG(H_temp, prior.R, prior.Rsolver, Omega, k)
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

# -log pi(theta), in this case independent uniform distributions on eta, delta, sigma
def neg_log_hyperprior(theta, hyp_pr_params):
    min_eta, max_eta, min_del, max_del, min_sig, max_sig = hyp_pr_params
    eta, delta, sigma = theta
    theta_prior = np.log(max_eta-min_eta) + np.log(max_del-min_del) + np.log(max_sig-min_sig)
    # set prior value to 0 outside the domain of the prior (neg log value to large)
    if eta < min_eta or eta > max_eta or delta < min_del or delta > max_del or sigma < min_sig or sigma > max_sig:
        theta_prior = 1e30
    return theta_prior

# -log pi(theta | y) (- log posterior marginal joint pdf of theta)
# warning: changes noise variance in problem.misfit for each theta
def neglogpi_theta(theta, lmbda, V, tol, neg_adj_y, pretheta, hyp_pr_params, problem, use_CG=False):
    preeta, predelta, presigma = pretheta
    eta, delta, sigma = theta

    # make copy of lmbda, V truncated to new theta
    cutoff = tol*sigma**2*delta**2/presigma**2/predelta**2
    r_cutoff = np.argmax(lmbda < cutoff)+1
    if r_cutoff == 1:
        r_cutoff = lmbda.size
        print("Approximation is too low rank, cutoff eigval not achieved")
    lmbda_new = lmbda[0:r_cutoff]
    V_new = MultiVector(V[0], r_cutoff)
    for i in range(r_cutoff):
        V_new[i][:] = V[i]

    # compute new posterior
    posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda_new, V_new, neg_adj_y, pretheta, problem, use_CG)
    
    # -log(|Q_pr|/|Q_post|)
    det_ratio = 0.0
    for ll in posterior.d:
        det_ratio += np.log(1+ll)
    # -log|Q_eps| = 2*n_obs*log(sigma)
    det_ratio += 2*targets.shape[0]*observation_times.shape[0]*np.log(sigma)
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

    preprior = BiLaplacianPrior(problem.Vh[PARAMETER], preeta*predelta, predelta, robin_bc=True)
    preprior.mean = dl.interpolate(dl.Constant(prior_mean), problem.Vh[PARAMETER]).vector()
    problem.prior = preprior
    problem.misfit.noise_variance = presigma**2
        
    H_misfit_only = ReducedHessian(problem, misfit_only=True)
    pad = int(k/2)
    Omega = MultiVector(x[PARAMETER], k+pad)
    parRandom.normal(1., Omega)

    lmbda, V = singlePassG(H_misfit_only, preprior.R, preprior.Rsolver, Omega, k) 
    return lmbda, V

# returns errors in posterior covariance for ranks ks, first rank at which error is below threshold,
# and low rank approximation with rank max(ks)
def PostCovError(theta, lmbda, V, neg_adj_y, pretheta, truth, ks, threshold, problem):
    errs = np.zeros(len(ks))
    first_ii = -1
    for ii in range(len(ks)):
        print(ii)
        k = ks[ii]
        posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda[0:k], mv_k(V,k), neg_adj_y, pretheta, problem)
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

# average function of a box in the domain
# handles 2d and 3d
class BoxAverage(dl.UserExpression):
    def __init__(self, boxlims, **kwargs):
        super().__init__(**kwargs)
        self.xmin = boxlims[0]; self.xmax = boxlims[1]
        self.ymin = boxlims[2]; self.ymax = boxlims[3]
        if boxlims.size == 6:
            self.zmin = boxlims[4]; self.zmax = boxlims[5]
    def eval(self, value, x):
        if boxlims.size == 4 and self.xmin < x[0] < self.xmax and self.ymin < x[1] < self.ymax:
            value[0] = 1.0/(xmax-xmin)/(ymax-ymin)
        elif boxlims.size == 6 and self.xmin < x[0] < self.xmax and self.ymin < x[1] < self.ymax and self.zmin < x[2] < self.zmax:
            value[0] = 1.0/(self.xmax-self.xmin)/(self.ymax-self.ymin)/(self.zmax-self.zmin)
        else:
            value[0] = 0.0
    def value_shape(self):
        return ()

# compute average of u0 over a box in the domain
def QoI(u0, Vh, boxlims):
    box_avg_expr = BoxAverage(boxlims)
    u0fun = dl.Function(Vh,u0)
    return dl.assemble(u0fun * box_avg_expr * dl.dx)

# apply adjoint of qoi to a scalar
# in this case, function that is 0 outside of box, averages to input value inside
def QoIadj(qoi, Vh, boxlims):
    box_avg_expr = BoxAverage(boxlims)
    m_test = dl.TestFunction(Vh)
    L_form = box_avg_expr * m_test * dl.dx
    b = dl.assemble(L_form)
    b *= qoi
    return b

# find distribution of QoI for fixed theta
def QoIdist_fixed_theta(qoi, theta, boxlims, lmbda, V, neg_adj_y, pretheta, problem):
    output = np.zeros(len(qoi))
    # find Gaussian pi(qoi|theta,y)
    prior = BiLaplacianPrior(Vh, theta[0]*theta[1], theta[1], robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(prior_mean), problem.Vh[PARAMETER]).vector()
    posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda, V, neg_adj_y, pretheta, problem)
    # mean
    mm = QoI(posterior.mean, problem.Vh[PARAMETER], boxlims)
    # var = QoI(Q_post_inv*QoIadj(1))
    temp = dl.Vector(posterior.prior.R.mpi_comm())
    posterior.prior.init_vector(temp,0)
    problem.prior = prior
    problem.misfit.noise_variance = theta[2]**2
    H = ReducedHessian(problem, misfit_only=False) 
    solver = CGSolverSteihaug()
    solver.set_operator(H)
    solver.set_preconditioner( posterior.Hlr )
    solver.parameters["print_level"] = -1
    solver.parameters["rel_tolerance"] = 1e-6
    b = QoIadj(1.0, problem.Vh[PARAMETER], boxlims)
    solver.solve(temp, b)
    vv = temp.inner(b)
    # evaluate Gaussian at each qoi value
    for ii in range(len(qoi)):
        output[ii] = np.exp(-(qoi[ii]-mm)**2/2/vv)/np.sqrt(2*np.pi*vv)
    return output

# return marginal distribution of QoI evaluated at a vector of qoi's
# (some day make this work for a single scalar qoi too)
def QoIdist(qoi, quad_points, pi_theta_quad, d_area, boxlims, lmbda, V, neg_adj_y, pretheta, problem):
    output = np.zeros(len(qoi))
    gauss_evals = np.zeros((len(qoi),quad_points.shape[0]))
    # for each quadrature point:
    for idx in range(quad_points.shape[0]):
        # find Gaussian pi(qoi|theta,y) where theta = the quadrature point
        theta = quad_points[idx,:]
        gauss_evals[:,idx] = QoIdist_fixed_theta(qoi, theta, boxlims, lmbda, V, neg_adj_y, pretheta, problem)
        # multiply by pi(theta|y) at qpt and area/volume element and add
        output += d_area*pi_theta_quad[idx]*gauss_evals[:,idx]
    return output

## QoI distribution for when QoI is pointwise evaluation
## Finds QoI distribution at a list of locations simultaneously (unlike general code above)
# find pi(x^i|y) for each location i in locs at values u_0_eval of u_0
def posterior_marginals(locs, u_0_eval, quad_points, pi_theta_quad, d_area, lmbda, V, neg_adj_y, pretheta, problem):
    output = np.zeros((len(locs),len(u_0_eval)))
    gauss_evals = np.zeros((len(locs),len(u_0_eval),quad_points.shape[0]))
    for idx in range(quad_points.shape[0]):
        theta = quad_points[idx,:]
        posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda, V, neg_adj_y, pretheta, problem)
        # pi(u_0^i|theta,y)
        posterior_var,pr,corr = posterior.pointwise_variance(method="Exact")
        mm = dl.Function(problem.Vh[PARAMETER],posterior.mean)
        vv = dl.Function(problem.Vh[PARAMETER],posterior_var)
        for ii in range(len(locs)):
            for jj in range(len(u_0_eval)):
                uu = u_0_eval[jj]
                gauss_evals[ii,jj,idx] = np.exp(-(uu-mm(locs[ii]))**2/2/vv(locs[ii]))/np.sqrt(2*np.pi*vv(locs[ii]))
        output += d_area*pi_theta_quad[idx]*gauss_evals[:,:,idx]
    return output,gauss_evals

# %%
dim = 2
# ******************** 2D Problem Setup ****************************
if dim == 2:
    ## Import 2D mesh
    # number of vertices in mesh (selects which mesh to import)
    # current options are 253, 399, 557, 605, 729, 1225, 1879, 2363, 2779, 5443, 8335, 11521
    verts = 2363
    mesh = dl.refine( dl.Mesh("meshes/adv_diff_dofs_{0}.xml".format(verts)) )
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh2 = dl.FunctionSpace(mesh, "Lagrange", 2)
    print("Number of elements: {0}".format(mesh.num_cells()))
    print("Number of dofs, first order: {0}".format(Vh.dim()))
    print("Number of dofs, second order: {0}".format(Vh2.dim()))

    ## Initial Condition
    ic_expr = dl.Expression(
        'std::min(0.5,std::exp(-100*(std::pow(x[0]-0.35,2) +  std::pow(x[1]-0.7,2))))',
        element=Vh2.ufl_element())
    true_initial_condition = dl.interpolate(ic_expr, Vh).vector()

    ## Advection velocity field
    wind_velocity = computeVelocityField(mesh)

    ## Observation points along building edges
    targets = np.loadtxt('targets/targets_2d.txt')

    # plot velocity field and target locations
    Xh = dl.VectorFunctionSpace(mesh,'Lagrange', 2)
    vh = dl.project(wind_velocity,Xh)
    nb.plot(vh)
    plt.scatter(targets[:,0],targets[:,1],color='red')

# ******************** 3D Problem Setup ****************************
elif dim == 3:
    ## Import 3D mesh and advection velocity field (precomputed)
    verts = 7480

    mesh = dl.Mesh()
    hdf = dl.HDF5File(mesh.mpi_comm(), "velocity_fields/velocity_field_{0}.h5".format(verts), "r")
    hdf.read(mesh, "/mesh", False)
    Xh = dl.VectorFunctionSpace(mesh,'Lagrange', 2)
    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh2 = dl.FunctionSpace(mesh, "Lagrange", 2)
    print("Number of elements: {0}".format(mesh.num_cells()))
    print("Number of dofs, first order: {0}".format(Vh.dim()))
    print("Number of dofs, second order: {0}".format(Vh2.dim()))

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
        element=Vh.ufl_element()
    )
    true_initial_condition = dl.interpolate(ic_expr, Vh).vector()

    ## Observation points
    targets = np.loadtxt('targets/targets_3d.txt')
else:
    print("Dimension must be 2 or 3")

# %% Solve Forward Problem
nt = 80
t_init         = 0.
t_final        = 4.
t_1            = 2.4
dt             = t_final/nt
observation_dt = 0.4
simulation_times = np.arange(t_init, t_final+.5*dt, dt)
observation_times = np.arange(t_1, t_final+.5*dt, observation_dt)
print(observation_times)

print ("Number of observation points: {0}".format(targets.shape[0]) )
print ("Number of observation times: {0}".format(observation_times.shape[0]) )
# initialize observations
misfit = SpaceTimePointwiseStateObservation(Vh2, observation_times, targets)

# %%
## Initialize Problem
# Set up prior (required by TimeDependentAD but not used)
eta = 0.125
delta = 8.
# covariance C = ((delta * (I - eta * Laplacian))^{-2}
prior = BiLaplacianPrior(Vh, eta*delta, delta, robin_bc=True)
prior_mean = 0.
prior.mean = dl.interpolate(dl.Constant(prior_mean), Vh).vector()

if dim == 2:
    kappa = 0.001
elif dim == 3:
    kappa = 0.003
problem_true = TimeDependentAD(mesh, [Vh2,Vh,Vh2], prior, misfit, simulation_times, kappa, wind_velocity, True)
sigma_true = 1e-2  # true noise stdev

#%%
# initialize vector in the state space
utrue = problem_true.generate_vector(STATE)
x = [utrue, true_initial_condition, None]
# solve forward problem
problem_true.solveFwd(x[STATE], x)
# observe solution and add error
misfit.observe(x, misfit.d)
parRandom.normal_perturb(sigma_true,misfit.d)
misfit.noise_variance = sigma_true**2

# # %%
# modelVerify(problem_true, prior.mean, is_quadratic=True)

# %% Plot/save forward solution
save_fwd_soln_3d = False
if dim == 2:
    ic_func = dl.Function(Vh)
    ic_func.vector()[:] = true_initial_condition
    ic_Vh2 = dl.project(ic_func, Vh2).vector()
    # nb.show_solution(Vh2, ic_Vh2, utrue, "Solution")
elif dim == 3 and save_fwd_soln_3d:
    # Create the PVD file
    file_pvd = dl.File("forward_sol_{0}.pvd".format(verts))
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

# %% Define parameters
# hyperprior parameters (independent, uniform in [min,max])
if dim == 2:
    min_eta = 0.0015
elif dim == 3:
    min_eta = 0.01
max_eta = 10
min_del = 1; max_del = 100
min_sig = 3e-3; max_sig = 1e-1
hyp_pr_params = np.array([min_eta, max_eta, min_del, max_del, min_sig, max_sig])
pretheta = np.array([min_eta, 1, 1])

# choose rank r by lambda < tol*sigma^2*delta^2
tol = 1e-2

# %% Precompute -A^T y in MAP point
# note -- problem contains the true noise variance in misfit. Will be overwritten in computation.
problem = TimeDependentAD(mesh, [Vh2,Vh,Vh2], prior, misfit, simulation_times, kappa, wind_velocity, True)
problem.misfit.noise_variance = 1
# This computes -A^T Q_eps y, so -A^T y with sigma = 1
[u0,m0,neg_adj_y] = problem.generate_vector()
problem.solveFwd(u0, [u0,m0,neg_adj_y])
problem.solveAdj(neg_adj_y, [u0,m0,neg_adj_y]) 

# %% Compute full spectrum of prior preconditioned update
one_spectrum_plot = False
if one_spectrum_plot:
    if dim == 2:
        theta3 = np.array([0.015, 12.5, 0.01])
    elif dim == 3:
        theta3 = np.array([0.08, 15, 0.01])
    # "true" posterior covariance trace, for error comparison
    r = 250
    lmbda_prior3, V_prior3 = LowRankApprox([theta3[0], theta3[1], theta3[2]], r, problem)

    # Plot spectrum
    fig = plt.figure(figsize=(10,7.2))
    plt.rcParams.update({'font.size': 20})
    plt.semilogy(range(len(lmbda_prior3)), lmbda_prior3*(theta3[2]**2)*(theta3[1]**2), linewidth=3, label=rf'$\eta=${theta3[0]}')
    plt.ylabel(r'$\lambda_i$')
    plt.xlabel(r'$i$')
    plt.legend()

#%% Saving spectra for dimension independence check
save_spectra_dim_check = False
if save_spectra_dim_check:
    header = "r \t\t 605linear \t\t 605quadratic \t\t 1225linear \t\t 1225quadratic \t\t 2363linear \t\t 2363quadratic \t\t 5443linear \t\t 5443quadratic \t\t 605cubic \t\t 1225cubic \t\t 2363cubic \t\t 5443cubic"
    file_path = "images/spectra_dim_indep.txt"
    # np.savetxt(file_path, np.column_stack((np.arange(1,r+1,1), lmbda_prior3)), delimiter="\t", header=header, fmt='%10.14f', comments="")
    existing_data = np.loadtxt(file_path, skiprows=1, delimiter="\t")
    updated_data = np.column_stack((existing_data, lmbda_prior3))
    np.savetxt(file_path, updated_data, delimiter="\t", header=header, fmt='%10.14f', comments="")

#%% Compute spectra of low-rank approx for 3 methods
full_spectra_plot = False
if full_spectra_plot:
    r = 250
    if dim == 2:
        theta1 = np.array([0.003, 50, 0.01])
        theta2 = np.array([0.0075, 25, 0.01])
    if dim == 3:
        theta1 = np.array([0.02, 60, 0.01])
        theta2 = np.array([0.04, 30, 0.01])
    lmbda_prior1, V_prior1 = LowRankApprox(theta1, r, problem)
    lmbda_prior2, V_prior2 = LowRankApprox(theta2, r, problem)
    lmbda_weak, V_weak = LowRankApprox(np.array([min_eta, 1.0, 1.0]), r, problem)
    lmbda_unprecon, V_unprecon = LowRankApprox(np.array([0.0, 1.0, 1.0]), r, problem)
    l_pr1_scaled = lmbda_prior1*(theta1[2]**2)*(theta1[1]**2)
    l_pr2_scaled = lmbda_prior2*(theta2[2]**2)*(theta2[1]**2)
    l_pr3_scaled = lmbda_prior3*(theta3[2]**2)*(theta3[1]**2)

    # Ranks for min cutoff
    min_cutoff = tol * min_sig**2 * min_del**2
    r_w_min = np.argmax(lmbda_weak < min_cutoff)
    r_u_min = np.argmax(lmbda_unprecon < min_cutoff)

    # Ranks for updated cutoff
    cutoff_3 = tol * theta3[1]**2 * theta3[2]**2
    r_p_3 = np.argmax(l_pr3_scaled < cutoff_3)
    r_w_3 = np.argmax(lmbda_weak < cutoff_3)
    r_u_3 = np.argmax(lmbda_unprecon < cutoff_3)
    print(f'min cutoff ranks: weak = {r_w_min}, unprecon = {r_u_min}')
    print(f'theta3 cutoff ranks: prior = {r_p_3}, weak = {r_w_3}, unprecon = {r_u_3}')

    fig = plt.figure(figsize=(10,7.2))
    plt.rcParams.update({'font.size': 20})
    plt.semilogy(range(r), lmbda_unprecon, linewidth=3, label=rf'$\eta=${0}')
    plt.semilogy(range(r), lmbda_weak, linewidth=3, label=rf'$\eta=${min_eta}')
    plt.semilogy(range(r), l_pr1_scaled, linewidth=3, label=rf'$\eta=${theta1[0]}')
    plt.semilogy(range(r), l_pr2_scaled, linewidth=3, label=rf'$\eta=${theta2[0]}')
    plt.semilogy(range(r), l_pr3_scaled[0:r], linewidth=3, label=rf'$\eta=${theta3[0]}')
    plt.ylabel(r'$\Lambda_{ii}$')
    plt.xlabel(r'$i$')
    plt.legend()

# %% Save spectrum data
save_spectra = False
if save_spectra:
    header = "r \t\t unprecon \t\t weakest \t\t prior1 \t\t prior2 \t\t prior3"
    np.savetxt("images/spectra_3DAD_full.txt", np.column_stack((np.arange(1,r+1,1), lmbda_unprecon, lmbda_weak, l_pr1_scaled, l_pr2_scaled, l_pr3_scaled[0:r])), delimiter="\t", header=header, fmt='%10.14f', comments="")

#%%
error_plot = False
if error_plot:
    # low rank approx
    theta = np.array([0.015, 12.5, 0.01])
    lmbda_pr, V_pr = LowRankApprox(theta, 250, problem)
    lmbda_weak, V_weak = LowRankApprox(pretheta, 250, problem)
    lmbda_un, V_un = LowRankApprox(np.array([0, 1, 1]), 250, problem)
    # true values
    posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda_pr, V_pr, neg_adj_y, theta, problem, use_CG=True)
    uQu = 0.5*mg.inner(posterior.mean)
    det_ratio = 0.0
    for ll in posterior.d:
        det_ratio += 0.5*np.log(1+ll)

    # ranks = np.linspace(11, 80, 24)
    ranks = np.linspace(11, 80, 70)
    e1_pr = np.zeros(len(ranks)); e2_pr = np.zeros(len(ranks))
    e1_weak = np.zeros(len(ranks)); e2_weak = np.zeros(len(ranks))
    e1_un = np.zeros(len(ranks)); e2_un = np.zeros(len(ranks))
    for idx in range(len(ranks)):
        r_trunc = int(ranks[idx])
        lmbda_trunc_pr = lmbda_pr[0:r_trunc-1]
        lmbda_trunc_weak = lmbda_weak[0:r_trunc-1]
        lmbda_trunc_un = lmbda_un[0:r_trunc-1]
        V_trunc_pr = MultiVector(V_pr[0], r_trunc-1)
        V_trunc_weak = MultiVector(V_weak[0], r_trunc-1)
        V_trunc_un = MultiVector(V_un[0], r_trunc-1)
        for i in range(r_trunc-1):
            V_trunc_pr[i][:] = V_pr[i]
            V_trunc_weak[i][:] = V_weak[i]
            V_trunc_un[i][:] = V_un[i]

        posteriorNoCG_pr,mgNoCG_pr,lmbda_new,V_new = ComputePosterior(theta, lmbda_trunc_pr, V_trunc_pr, neg_adj_y, theta, problem)
        e1_pr[idx] = det_ratio
        for ll in posteriorNoCG_pr.d:
            e1_pr[idx] -= 0.5*np.log(1+ll)
        e2_pr[idx] = uQu-0.5*mgNoCG_pr.inner(posteriorNoCG_pr.mean)

        posteriorNoCG_weak,mgNoCG_weak,lmbda_new,V_new = ComputePosterior(theta, lmbda_trunc_weak, V_trunc_weak, neg_adj_y, pretheta, problem)
        e1_weak[idx] = det_ratio
        for ll in posteriorNoCG_weak.d:
            e1_weak[idx] -= 0.5*np.log(1+ll)
        e2_weak[idx] = uQu-0.5*mgNoCG_weak.inner(posteriorNoCG_weak.mean)

        posteriorNoCG_un,mgNoCG_un,lmbda_new,V_new = ComputePosterior(theta, lmbda_trunc_un, V_trunc_un, neg_adj_y, np.array([0,1,1]), problem)
        e1_un[idx] = det_ratio
        for ll in posteriorNoCG_un.d:
            e1_un[idx] -= 0.5*np.log(1+ll)
        e2_un[idx] = uQu-0.5*mgNoCG_un.inner(posteriorNoCG_un.mean)
    plt.semilogy(ranks, e1_pr, color="green", label="e1 PP")
    plt.semilogy(ranks, e2_pr, color="green", linestyle="--", label="e2 PP")
    plt.semilogy(ranks, e1_weak, color="orange", label="e1 WP")
    plt.semilogy(ranks, e2_weak, color="orange", linestyle="--", label="e2 WP")
    plt.semilogy(ranks, e1_un, color="blue", label="e1 UP")
    plt.semilogy(ranks, e2_un, color="blue", linestyle="--", label="e2 UP")
    plt.legend()
    plt.xlabel('rank')
    plt.ylabel('error')

    # header = "r \t\t e1_pr \t\t e1_weak \t\t e1_un \t\t e2_pr \t\t e2_weak \t\t e2_un"
    # np.savetxt("images/log_pi_error.txt", np.column_stack((ranks, e1_pr, e1_weak, e1_un, e2_pr, e2_weak, e2_un)), delimiter="\t", header=header, fmt='%10.14f', comments="")

    cutoff = 1e-2
    r_p_err = int(ranks[0]) + np.argmax((e1_pr+e2_pr) < cutoff)+1
    r_w_err = int(ranks[0]) + np.argmax((e1_weak+e2_weak) < cutoff)+1
    r_u_err = int(ranks[0]) + np.argmax((e1_un+e2_un) < cutoff)+1
    print(r_p_err, r_w_err, r_u_err)

# nb.plot(dl.Function(Vh,posterior.mean))
# true_trace,pr_tr,corr_tr = posterior.trace(method="Exact")

# %% Find low rank approx of prior-to-posterior update
if dim == 2:
    r_p = 50 ; r_w = 95; r_u = 110
elif dim == 3:
    r_p = 100; r_w = 199; r_u = 237
precon = 'unprecon'

if precon == 'unprecon':
    pretheta[0] = 0.0
    r = r_u
elif precon == 'prior':
    r = r_p
else:
    assert precon == 'weakest', 'precon should have value prior, unprecon, or weakest'
    r = r_w

# %% Compute -log pi for a range of thetas
%%prun 
if precon == 'weakest' or precon == 'unprecon':
    lmbda, V = LowRankApprox(pretheta, r, problem)
#%%
%%prun
ne = 1
nd = 1
ns = 1
if dim == 2:
    eta_range = np.linspace(0.0015,0.02,ne)
    d_range = np.linspace(15,60,nd)
    s_range = np.linspace(8.5e-3, 1.2e-2, ns)
elif dim == 3:
    eta_range = np.linspace(0.01,0.2,ne)
    d_range = np.linspace(2,80,nd)
    s_range = np.linspace(10e-3, 1e-2, ns)
logpi = np.zeros((ne,nd,ns))
print('Progress in indices computed from (0,0,0) to ({0},{1},{2}):'.format(ne-1,nd-1,ns-1))
for i in range(len(eta_range)):
    for j in range(len(d_range)):
        for k in range(len(s_range)):
            #compute -log pi_hat(theta)
            theta = np.array([eta_range[i],d_range[j],s_range[k]])
            if precon == 'prior':
                pretheta = theta.tolist()
                lmbda, V = LowRankApprox(pretheta, r, problem)
            logpi[i,j,k] = neglogpi_theta(theta, lmbda, V, tol, neg_adj_y, pretheta, hyp_pr_params, problem, use_CG=False)
            print('({0},{1},{2})'.format(i,j,k))

#%%
pitheta = np.exp(-logpi+np.min(logpi))
# #%%
# etamesh,dmesh,smesh = np.meshgrid(eta_range, d_range, s_range, indexing='ij')
# header = "eta \t\t delta \t\t sigma \t\t pi_theta"
# np.savetxt("images/pi_theta_40x40x20.txt", np.column_stack((etamesh.ravel(), dmesh.ravel(), smesh.ravel(), pitheta.ravel())), delimiter="\t", header=header, fmt=('%g', '%g', '%g', '%e'), comments="")
# %%

sig_idx = 0
# scaled arbitrarily to have max value 1 in order to avoid overflow errors
fig = plt.figure(figsize=(10,7.2))
plt.rcParams.update({'font.size': 16})
plt.set_cmap('bone')
plt.pcolormesh(d_range,eta_range,pitheta[:,:,sig_idx])
plt.colorbar()
# plt.title(r'$\pi(\eta, \delta | y), dofs={0}$'.format(dofs))
plt.title(r'$\pi(\eta, \delta, \sigma | y)$')
plt.ylabel(r'$\eta$')
plt.xlabel(r'$\delta$')
# plt.savefig("pi_gamma_delta.pdf",bbox_inches='tight', pad_inches=0)

# %%
eta_idx = 0; d_idx = 0
fig = plt.figure(figsize=(10,7.2))
plt.rcParams.update({'font.size': 16})
plt.plot(s_range,pitheta[eta_idx,d_idx,:])
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$\pi(\eta, \delta, \sigma|y)$')

# %%

# scaled arbitrarily to have max value 1 in order to avoid overflow errors
fig = plt.figure(figsize=(10,7.2))
plt.rcParams.update({'font.size': 16})
plt.set_cmap('bone')
plt.pcolormesh(s_range,eta_range,pitheta[:,d_idx,:])
plt.colorbar()
# plt.title(r'$\pi(\eta, \delta | y), dofs={0}$'.format(dofs))
plt.title(rf'$\pi(\eta, \delta, \sigma | y), \: \delta = {d_range[d_idx]:.2f}$')
plt.ylabel(r'$\eta$')
plt.xlabel(r'$\sigma$')
# plt.savefig("pi_gamma_delta.pdf",bbox_inches='tight', pad_inches=0)

# %%
## Compute MAP point of pi(theta | y)

opt_start = time.time()
def neglogpi_helper(theta):
    return neglogpi_theta(theta, lmbda, V, tol, neg_adj_y, pretheta, hyp_pr_params, problem)
def opt_callback(intermediate_result):
    # print(f"Iteration: {intermediate_result.nit}")
    print(f"Current x: {intermediate_result.x}")
    print(f"Objective value: {intermediate_result.fun}")
theta0 = np.array([1, 10, 0.05])
theta_opt = opt.minimize(neglogpi_helper,theta0,method='Nelder-Mead',callback=opt_callback, options={'disp':True,'xatol':1e-2,'fatol':1e-2})
opt_end = time.time()
print(f"Optimization time: {opt_end-opt_start} seconds")

theta_MAP = theta_opt.x
print(f"MAP point of pi(theta|y): {theta_MAP}")
# theta_MAP = np.array([3.45302293e-02, 3.39541763e+01, 9.75111448e-03])

# %%
## Find inverse Hessian at MAP point

# choosing the finite difference dx's here is finicky -- can't be much smaller
if dim == 2:
    dtheta = [1e-3,8e-1,1e-5]
elif dim == 3:
    dtheta = [1e-3,2,1e-5]
ntheta = np.size(dtheta)

Hess_MAP = np.zeros((ntheta,ntheta))

# compute necessary function evaluations for Hessian finite difference estimation
neglogpiMAP = neglogpi_helper(theta_MAP)
plustwo = np.zeros((ntheta,ntheta))
plusone = np.zeros(ntheta)
minusone = np.zeros(ntheta)
for i in range(ntheta):
    dtheta_i = np.zeros(ntheta)
    dtheta_i[i] = dtheta[i]
    plusone[i] = neglogpi_helper(theta_MAP + dtheta_i)
    minusone[i] = neglogpi_helper(theta_MAP - dtheta_i)
    for j in range(i+1,ntheta):
        dtheta_i_j = np.zeros(ntheta)
        dtheta_i_j[i] = dtheta[i]; dtheta_i_j[j] = dtheta[j]
        plustwo[i,j] = neglogpi_helper(theta_MAP + dtheta_i_j)
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
Hinv_L_sqrt = np.diag(np.sqrt(Hinv_lam))
print(Hinv_L_sqrt)
def theta_of_z(z):
    return theta_MAP + np.dot(Hinv_V,np.dot(Hinv_L_sqrt,z))

# %%
# for each coordinate of z, find its values with significant probability
if dim == 2:
    delta_z = 1
    delta_pi = 2.5
if dim == 3:
    delta_z = 0.8
    delta_pi = 3
maxiter = 20

z_highprob = [np.array([0.0]) for i in range(ntheta)]
for idx in range(ntheta):
    z = np.zeros(ntheta)
    z[idx] = delta_z
    count = 0
    while all(theta_of_z(z)>0) and neglogpi_helper(theta_of_z(z)) - neglogpiMAP < delta_pi and count < maxiter:
        z_highprob[idx] = np.append(z_highprob[idx],z[idx])
        z[idx] += delta_z
        count += 1
        print(count)
    count = 0
    z[idx] = -delta_z
    while all(theta_of_z(z)>0) and neglogpi_helper(theta_of_z(z)) - neglogpiMAP < delta_pi and count < maxiter:
        z_highprob[idx] = np.append(z_highprob[idx],z[idx])
        z[idx] -= delta_z
        count += 1
        print(count)

# %%
# Find quadrature points

# list pairs of points given two lists of point locations
# e.g., inputs [[0 0],[1 1]] and [2 3], output [[0 0 2],[1 1 2],[0 0 3],[1 1 3]]
# first input must be list of lists, second must be list
def pt_pairs(list1, list2):
    newlist = []
    for i in range(len(list1)):
        for j in range(len(list2)):
            newlist.append(list1[i]+[list2[j]])
    return newlist

quad_start = time.time()
# use to recursively find all combinations of possible points
all_points = [[zval] for zval in z_highprob[0]]
if ntheta > 1:
    for idx in range(1,ntheta):
        all_points = pt_pairs(all_points, z_highprob[idx])

# search through them for only the ones with high enough probability
# could be more efficient -- don't recalculate along axes, and/or store values for later
quad_points = []
print('Points to be checked: {0}'.format(len(all_points)))
for i in range(len(all_points)):
    print(i)
    theta_i = theta_of_z(np.array(all_points[i]))
    if (theta_i[0] > min_eta and theta_i[1] > min_del and theta_i[2] > min_sig and 
        theta_i[0] < max_eta and theta_i[1] < max_del and theta_i[2] < max_sig):
        is_valid_point = True
    else:
        is_valid_point = False
        print("found invalid point")
    if is_valid_point and neglogpi_helper(theta_i) - neglogpiMAP < delta_pi:
        quad_points.append(theta_i)
quad_points = np.array(quad_points)
quad_end = time.time()
print(f"Quad points checking time: {quad_end-quad_start} seconds")

# %%
# scatter plot of quadrature points
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(quad_points[:,0],quad_points[:,1],quad_points[:,2],color='black',label='quadrature points')
if dim==3:
    ax.scatter(0.02, 60, 0.01, color='green') 
    ax.scatter(0.04, 30, 0.01, color='green') 
    ax.scatter(0.08, 15, 0.01, color='green') 
ax.scatter(theta_MAP[0], theta_MAP[1], theta_MAP[2], color='red') 
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\delta$')
ax.set_zlabel(r'$\sigma$')
ax.set_title('Quadrature Points')

#%% Saving (scaled) quadrature points for plot
# quad_pts_scaled = quad_points.copy()
# quad_pts_scaled[:,0] = quad_pts_scaled[:,0]*2000
# quad_pts_scaled[:,2] = quad_pts_scaled[:,2]*10000
# header = "eta, delta, sigma"
# np.savetxt("images/quad_points.txt", quad_points, delimiter=",", header=header, fmt='%10.10f', comments="")

#%%
# precompute pi(theta|y) at quad points and scale to integrate to 1
# (if not increasing resolution, could store these from earlier)
pi_theta_quad = np.zeros(quad_points.shape[0])
for i in range(quad_points.shape[0]):
    pi_theta_quad[i] = np.exp(-neglogpi_helper(quad_points[i,:])+neglogpiMAP)

# find Z such that 1/Z*pi(theta|y) integrates to ~1 using quadrature
d_area = np.sqrt(np.prod(Hinv_lam))
Z = np.sum(pi_theta_quad)*d_area
# scale evaluations of pi(theta|y)
pi_theta_quad = pi_theta_quad/Z

#%%
### Compute marginal distribution of QoI that is a linear scalar function of u_0
## QoI here is the average of u_0 in part of the domain

# box location
if dim == 2:
    xmin = 0.2; xmax = 0.4
    ymin = 0.7; ymax = 0.9
    boxlims = np.array([xmin, xmax, ymin, ymax])
    fig, ax = plt.subplots()
    nb.plot(dl.Function(Vh,true_initial_condition),mytitle='IC and Box Location')
    rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
elif dim == 3:
    xmin = 0.15; xmax = 0.3
    ymin = 0.7; ymax = 0.85
    zmin = 0.5; zmax = 0.65
    boxlims = np.array([xmin, xmax, ymin, ymax])


# print QoI(constant 1 function) to test error introduced by finite element approx
testu0 = dl.interpolate(dl.Constant(1), Vh2).vector()
print(f"QoI(constant 1 function) = {QoI(testu0, Vh2, boxlims)}")

# %%
# evaluate pi(qoi|y) at range of qoi values
qoi_range = np.linspace(0.1,0.275,100)
pi_qoi = QoIdist(qoi_range, quad_points, pi_theta_quad, d_area, boxlims, lmbda, V, neg_adj_y, pretheta, problem)

#%%
theta_true = theta_MAP
theta1 = np.array([0.003, 50, 0.01])
theta3 = np.array([0.015, 12.5, 0.01])
pi_qoi_th_true = QoIdist_fixed_theta(qoi_range, theta_true, boxlims, lmbda, V, neg_adj_y, pretheta, problem)
pi_qoi_th_1 = QoIdist_fixed_theta(qoi_range, theta1, boxlims, lmbda, V, neg_adj_y, pretheta, problem)
pi_qoi_th_3 = QoIdist_fixed_theta(qoi_range, theta3, boxlims, lmbda, V, neg_adj_y, pretheta, problem)

# true QoI
true_QoI = QoI(true_initial_condition, Vh, boxlims)

#%%
# plot distribution of QoI
plt.figure(figsize=(10,7.2))
plt.rcParams.update({'font.size': 20})
plt.plot(qoi_range,pi_qoi_th_true,linewidth=3,color='green', label=rf"$\theta^\ast$")
plt.plot(qoi_range,pi_qoi_th_1,linewidth=3,color='red', label=rf"$\theta_1$")
plt.plot(qoi_range,pi_qoi_th_3,linewidth=3,color='orange', label=rf"$\theta_3$")
plt.plot(qoi_range,pi_qoi,linewidth=3,color='black', label=r"marginalized")
plt.axvline(x=true_QoI, color='black', linestyle="-.", label=r"true qoi")
plt.title(rf'Posterior Distribution of QoI $q$')
plt.ylabel(r"$\pi(q|y)$")
plt.xlabel(r"$q$")
plt.tight_layout()
plt.legend()

# #%%
# # Save data
# header = "q \t\t theta_opt \t\t theta_1 \t\t theta_3 \t\t marginalized"
# np.savetxt("images/piQoI.txt", np.column_stack((qoi_range, pi_qoi_th_true, pi_qoi_th_1, pi_qoi_th_3, pi_qoi)), delimiter="\t", header=header, fmt='%10.8f', comments="")
# %%
u_range = np.linspace(0.0,0.55,100)
locations = [[0.3,0.7],[0.45,0.55]]
true_u0_fun = dl.Function(Vh,true_initial_condition)
pi_u_0_i_evals,gauss_evals = posterior_marginals(locations, u_range, quad_points, pi_theta_quad, d_area, lmbda, V, neg_adj_y, pretheta)
# %%
pi_u_0_i_evals_norms = trapezoid(pi_u_0_i_evals,dx=u_range[1]-u_range[0],axis=1)
pi_u_0_i_evals_normalized = (pi_u_0_i_evals.T/pi_u_0_i_evals_norms).T
for idx in range(quad_points.shape[0]):
    gauss_evals[:,:,idx] = (gauss_evals[:,:,idx].T/pi_u_0_i_evals_norms).T
header = "x \t\t u_0_1 \t\t u_0_2"
# np.savetxt("images/pi_u_0.txt", np.column_stack((u_range, pi_u_0_i_evals_normalized.T)), delimiter="\t", header=header, fmt='%10.8f', comments="")

# %%
for ii in range(len(locations)):
    plt.figure(figsize=(10,4.5))
    plt.rcParams.update({'font.size': 16})
    plt.plot(u_range,pi_u_0_i_evals_normalized[ii,:],linewidth=3,color='green')
    plt.axvline(x=true_u0_fun(locations[ii]), color='purple', linestyle="-.", label=r"true $u_0^i$")
    #plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.title(f'Posterior Marginal Initial Condition, location {locations[ii]}')
    plt.ylabel(r"$\log \pi(u_0^i|y)$")
    plt.xlabel(r"$u_0$")
    plt.tight_layout()
    plt.legend()
    #plt.savefig("u_0_marginal.pdf")

# %%
xy = mesh.coordinates()

fig = plt.figure(figsize=(10,7.2))
im = nb.plot(dl.Function(Vh,true_initial_condition),mytitle='True Initial Condition',colorbar=False)
plt.plot(locations[0][0],locations[0][1],'ro',markersize=8,label=r'$x_1$') 
#plt.plot(locations[1][0],locations[1][1],'rx') 
plt.plot(locations[1][0],locations[1][1],'rs',markersize=10,label=r'$x_2$') 
plt.legend(loc='upper right', bbox_to_anchor=(0.9, 0.3))
fig.colorbar(im, pad=0.05)
# plt.savefig("point_locations.pdf",bbox_inches='tight', pad_inches=0)


 # %%
