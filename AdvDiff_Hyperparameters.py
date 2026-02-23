# %%

import dolfin as dl
#import ufl
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.integrate import trapezoid
#%matplotlib inline
import sys
import os
os.environ["HIPPYLIB_BASE_DIR"] = '/mnt/c/Users/Sonia/Documents/Courant/Research/INLA/PDE_INLA/hippylib'
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR') )
from hippylib import *
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR') + "/applications/ad_diff/" )
from model_ad_diff import SpaceTimePointwiseStateObservation, TimeDependentAD

# modified hippylib code
# model_ad_diff makes kappa no longer hardcoded
# posterior adds version for unpreconditioned low rank decomp
# prior changes Krylov solvers to LU/Cholesky for speed

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
#logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

import time
import line_profiler
#%load_ext line_profiler
np.random.seed(42)

# %%

# All from original AdvectionDiffusionBayesian tutorial
def v_boundary(x,on_boundary):
    return on_boundary

def q_boundary(x,on_boundary):
    return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS
        
def computeVelocityField(mesh):
    Xh = dl.VectorFunctionSpace(mesh,'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = dl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(1e2)
    
    g = dl.Expression(('0.0','(x[0] < 1e-14) - (x[0] > 1 - 1e-14)'), degree=1)
    bc1 = dl.DirichletBC(XW.sub(0), g, v_boundary)
    
    bc2 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')
    bcs = [bc1, bc2]
    
    vq = dl.Function(XW)
    (v,q) = dl.split(vq)
    (v_test, q_test) = dl.TestFunctions (XW)
    
    def strain(v):
        return dl.sym(dl.grad(v))
    
    F = ( (2./Re)*dl.inner(strain(v),strain(v_test))+ dl.inner (dl.nabla_grad(v)*v, v_test)
           - (q * dl.div(v_test)) + ( dl.div(v) * q_test) ) * dl.dx
           
    dl.solve(F == 0, vq, bcs, solver_parameters={"newton_solver":
                                         {"relative_tolerance":1e-4, "maximum_iterations":100}})
    
#     plt.figure(figsize=(15,5))
#     vh = dl.project(v,Xh)
#     qh = dl.project(q,Wh)
#     nb.plot(nb.coarsen_v(vh), subplot_loc=121,mytitle="Velocity")
#     nb.plot(qh, subplot_loc=122,mytitle="Pressure")
#     plt.show()
        
    return v

class Model():
    def __init__(self, mesh, Vh, misfit, simulation_times, kappa, wind_velocity):
        self.mesh = mesh
        self.Vh = Vh
        self.misfit = misfit
        self.simulation_times = simulation_times
        self.kappa = kappa
        self.wind_velocity = wind_velocity

# Helper function for slicing multivectors
def mv_k(mv, n):
    mv_n = MultiVector(mv[0], n)
    for i in range(n):
        mv_n[i].zero()
        mv_n[i].axpy(1.0, mv[i])
    return mv_n

def ComputePosterior(theta, lmbda, V, pretheta, model):
    '''
    Solve inverse problem
    Output: posterior object and mg = mu_u0^T Q_u0 + y^T Q_eps A
    Input:  theta = gamma, delta: hyperparameters of prior, lam: noise hyperparameter
            lmbda, V: low rank decomp of Q_pre^-1/2 A^T A Q_pre^-1/2, where Q_pre is a preconditioning prior precision
            pretheta = pregamma, predelta: parameters of Q_pre (or "none" if no Q_pre preconditioner), and prelam: noise precision used in low rank approx
            model = mesh, Vh, misfit, simulation_times, kappa, and wind_velocity
    '''
    pregamma, predelta, prelam = pretheta
    gamma, delta, lam = theta
    
    prior = BiLaplacianPrior(model.Vh, gamma, delta, robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(prior_mean), model.Vh).vector()

    problem = TimeDependentAD(model.mesh, [model.Vh,model.Vh,model.Vh], prior, model.misfit, model.simulation_times, model.kappa, model.wind_velocity, True)
    problem.misfit.noise_variance = 1/np.sqrt(lam)**2

    ## Compute the gradient
    [u,m,p] = problem.generate_vector()
    # forward solve
    problem.solveFwd(u, [u,m,p])
    # adjoint solve
    problem.solveAdj(p, [u,m,p]) # these last three lines don't use the prior and could be precomputed
    # initialize a vector in the parameter space
    mg = problem.generate_vector(PARAMETER)
    # evaluate gradient and store in mg
    grad_norm = problem.evalGradientParameter([u,m,p], mg) # this involves the prior so we should compute it here
    
    ## Compute posterior precision
    # matrix free application of posterior precision/covariance
    H = ReducedHessian(problem, misfit_only=True) 
    
    # no preconditioning
    if pregamma is None or predelta is None:
        precon = False
        lmbda_new = lmbda
        V_new = V
    # prior preconditioning
    elif pregamma == gamma and predelta == delta:
        precon = True
        lmbda_new = lmbda
        V_new = V
    # weakest preconditioning
    else:
        preprior = BiLaplacianPrior(model.Vh, pregamma, predelta, robin_bc=True)
        preprior.mean = dl.interpolate(dl.Constant(prior_mean), model.Vh).vector()
        # apply sqrt precon prior precision and sqrt inverse of prior precision
        Wtemp = MultiVector(V)
        Wtemp2 = MultiVector(V)
        W = MultiVector(V)
        for i in range(V.nvec()):
            preprior.A.mult(V[i],Wtemp[i]) # modifies Wtemp
            preprior.Msolver.solve(Wtemp2[i],Wtemp[i]) # modifies Wtemp2
            preprior.A.mult(Wtemp2[i],W[i]) # modifies W
        H_temp = LowRankOperator(lmbda, W)
        k = V.nvec()
        pad = 20 
        Omega = MultiVector(x[PARAMETER], k+pad)
        parRandom.normal(1., Omega)
        lmbda_new, V_new = singlePassG(H_temp, prior.R, prior.Rsolver, Omega, k)
        precon = True
    # correcting for noise precision used in low rank approx (prelam)
    lmbda_new = lmbda_new*lam/prelam
    posterior = GaussianLRPosterior(prior, lmbda_new, V_new, precon)

#     Compute posterior mean
    H.misfit_only = False
    solver = CGSolverSteihaug()
    solver.set_operator(H) # use lmbda, V plus the prior here?
    solver.set_preconditioner( posterior.Hlr )
    solver.parameters["print_level"] = -1
    solver.parameters["rel_tolerance"] = 1e-6
    solver.solve(m, -mg)
    problem.solveFwd(u, [u,m,p])
    posterior.mean = m
    
#     H = posterior.Hlr
#     H.solve(m,-mg)
#     posterior.mean = m

#     nb.plot(dl.Function(Vh,m))
    
    return posterior,mg,lmbda_new,V_new

# -log pi(theta | y) (- log posterior marginal joint pdf of theta)
# warning: changes noise variance in model.misfit for each theta
def neglogpi_theta(theta, lmbda, V, pretheta, hyp_pr_params, model):
    pregamma, predelta, prelam = pretheta
    gamma, delta, lam = theta
    alpha_del, alpha_gam, alpha_lam, beta_del, beta_gam, beta_lam, max_del, min_gam, max_lam = hyp_pr_params
    # compute new posterior
    model.misfit.noise_variance = 1/np.sqrt(lam)**2 # careful, model is mutable
    posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda, V, pretheta, model)
    
    # -log(|Q_pr|/|Q_post|)
    if pregamma is None or predelta is None:
        # Bhelp = Q_pr^-1 * V
        Bhelp = MultiVector(V_new)
        for i in range(V_new.nvec()):
            posterior.prior.Rsolver.solve(Bhelp[i],V_new[i])
        # B = diag(lmbda) * V^T Q_pr^-1 V + I
        B = V_new.dot_mv(Bhelp)
        for i in range(B.shape[0]):
            B[i,:] *= lmbda_new[i]
            B[i,i] += 1.
        Bvals = np.linalg.eigvals(B)
        det_ratio = np.sum(np.log(Bvals))
    else:
        det_ratio = 0.0
        for ll in posterior.d:
            det_ratio += np.log(1+ll)
    # -log|Q_eps| = n_obs*log(lam)
    det_ratio += -targets.shape[0]*observation_times.shape[0]*np.log(lam)
    det_ratio *= 0.5

    # -log pdf of gamma, delta, lambda prior
    theta_prior = - (alpha_del-1)*np.log(delta) - (alpha_gam-1)*np.log(gamma) + beta_del*delta + beta_gam*gamma - (alpha_lam-1)*np.log(lam) + beta_lam*lam
    # set prior value to 0 outside the domain of the prior (neg log value to large)
    if gamma < min_gam or delta > max_del or lam > max_lam:
        theta_prior = 1e30

    # -mu_post^T Q_post mu_post 
    uQu = 0.5*mg.inner(posterior.mean)

    # mu_pr^T Q_pr mu_pr
    Qmu = dl.Vector(posterior.prior.R.mpi_comm())
    posterior.prior.init_vector(Qmu,0)
    posterior.prior.R.mult(posterior.prior.mean,Qmu)
    muQmu = 0.5*posterior.prior.mean.inner(Qmu)

    # y^T Q_eps y
    yQy = 0.5*model.misfit.d.inner(model.misfit.d)*lam

    return det_ratio + theta_prior + uQu + muQmu + yQy

# find rank k approx to Hessian preconditioned by prior with params pretheta
# no prior preconditioning if pretheta[0] and pretheta[1] are None
def LowRankApprox(pretheta, k, model):
    # always compute low rank approx using largest possible noise precision lam
    pregamma, predelta, prelam = pretheta
    model.misfit.noise_variance = 1/np.sqrt(prelam)**2

    if pregamma is None or predelta is None:
        preprior = BiLaplacianPrior(model.Vh, 1., 8., robin_bc=True) # prior is required but not used
    else:
        preprior = BiLaplacianPrior(model.Vh, pregamma, predelta, robin_bc=True)
    preprior.mean = dl.interpolate(dl.Constant(prior_mean), Vh).vector()
    problem = TimeDependentAD(model.mesh, [model.Vh,model.Vh,model.Vh], preprior, model.misfit, model.simulation_times, model.kappa, model.wind_velocity, True)
        
    H_misfit_only = ReducedHessian(problem, misfit_only=True)
    pad = 20
    Omega = MultiVector(x[PARAMETER], k+pad)
    parRandom.normal(1., Omega)

    if pregamma is None and predelta is None:
        lmbda, V = singlePass(H_misfit_only, Omega, k)
    else:
        lmbda, V = singlePassG(H_misfit_only, preprior.R, preprior.Rsolver, Omega, k) 
    return lmbda, V

# returns errors in posterior covariance for ranks ks, first rank at which error is below threshold,
# and low rank approximation with rank max(ks)
def PostCovError(theta, pretheta, truth, truth_norm, ks, threshold, model):
    lmbda, V = LowRankApprox(pretheta, max(ks), model)
    errs = np.zeros(len(ks))
    first_ii = -1
    for ii in range(len(ks)):
        k = ks[ii]
        posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda[0:k], mv_k(V,k), pretheta, model)
        posterior_var,pr,corr = posterior.pointwise_variance(method="Exact")
        posterior.prior.M.mult(posterior_var,Mvar)
        diff = posterior_var - truth
        Mdiff = dl.Vector(posterior.prior.R.mpi_comm())
        posterior.prior.init_vector(Mdiff,0)
        posterior.prior.M.mult(diff,Mdiff)
        errs[ii] = diff.inner(Mdiff)/truth_norm
        # first rank with error below threshold
        if first_ii < 0 and errs[ii] < threshold:
            first_ii = ii
    if first_ii > 0:
        min_k = ks[first_ii]
    else:
        min_k = None
        print(f'did not reach threshold before rank {max(ks)}')
    return errs, min_k, lmbda, V

# average function of a box in the domain
class BoxAverage(dl.UserExpression):
    def __init__(self, boxlims, **kwargs):
        super().__init__(**kwargs)
        self.xmin = boxlims[0]; self.xmax = boxlims[1]
        self.ymin = boxlims[2]; self.ymax = boxlims[3]
    def eval(self, value, x):
        if xmin < x[0] < xmax and ymin < x[1] < ymax:
            value[0] = 1.0/(xmax-xmin)/(ymax-ymin)
        else:
            value[0] = 0.0
    def value_shape(self):
        return ()

# compute average of u0 over a box in the domain
def QoI(u0, Vh, boxlims):
    box_avg_expr = BoxAverage(boxlims)
    vec = dl.interpolate(box_avg_expr, Vh)
    u0fun = dl.Function(Vh,u0)
    qoi = dl.assemble(dl.inner(u0fun, vec) * dl.dx)
    return qoi 

# apply adjoint of qoi to a scalar
# in this case, function that is 0 outside of box, averages to input value inside
def QoIadj(qoi, Vh, boxlims):
    box_avg_expr = BoxAverage(boxlims)
    vec = dl.interpolate(box_avg_expr, Vh)
    vec.vector()[:] = vec.vector()[:]*qoi
    return vec.vector()

# find distribution of QoI for fixed theta
def QoIdist_fixed_theta(qoi, theta, boxlims, lmbda, V, pretheta, model):
    output = np.zeros(len(qoi))
    # find Gaussian pi(qoi|theta,y)
    prior = BiLaplacianPrior(Vh, theta[0], theta[1], robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(prior_mean), Vh).vector()
    posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda, V, pretheta, model)
    # mean
    mm = QoI(posterior.mean, Vh, boxlims)
    # var = QoI(Q_post_inv*QoIadj(1))
    temp = dl.Vector(posterior.prior.R.mpi_comm())
    posterior.prior.init_vector(temp,0)
    posterior.Hlr.solve(temp,QoIadj(1, Vh, boxlims))
    vv = QoI(temp, Vh, boxlims)
    # evaluate Gaussian at each qoi value
    for ii in range(len(qoi)):
        output[ii] = np.exp(-(qoi[ii]-mm)**2/2/vv)/np.sqrt(2*np.pi*vv)
    return output

# return marginal distribution of QoI evaluated at a vector of qoi's
# (some day make this work for a single scalar qoi too)
def QoIdist(qoi, quad_points, pi_theta_quad, d_area, boxlims, lmbda, V, pretheta, model):
    output = np.zeros(len(qoi))
    gauss_evals = np.zeros((len(qoi),quad_points.shape[0]))
    # for each quadrature point:
    for idx in range(quad_points.shape[0]):
        # find Gaussian pi(qoi|theta,y) where theta = the quadrature point
        theta = quad_points[idx,:]
        gauss_evals[:,idx] = QoIdist_fixed_theta(qoi, theta, boxlims, lmbda, V, pretheta, model)
        # multiply by pi(theta|y) at qpt and area/volume element and add
        output += d_area*pi_theta_quad[idx]*gauss_evals[:,idx]
    return output

## QoI distribution for when QoI is pointwise evaluation
## Finds QoI distribution at a list of locations simultaneously (unlike general code above)
# find pi(x^i|y) for each location i in locs at values u_0_eval of u_0
def posterior_marginals(locs, u_0_eval, quad_points, pi_theta_quad, d_area, lmbda, V, pretheta, model):
    output = np.zeros((len(locs),len(u_0_eval)))
    gauss_evals = np.zeros((len(locs),len(u_0_eval),quad_points.shape[0]))
    for idx in range(quad_points.shape[0]):
        theta = quad_points[idx,:]
        posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda, V, pretheta, model)
        # pi(u_0^i|theta,y)
        posterior_var,pr,corr = posterior.pointwise_variance(method="Exact")
        mm = dl.Function(Vh,posterior.mean)
        vv = dl.Function(Vh,posterior_var)
        for ii in range(len(locs)):
            for jj in range(len(u_0_eval)):
                uu = u_0_eval[jj]
                gauss_evals[ii,jj,idx] = np.exp(-(uu-mm(locs[ii]))**2/2/vv(locs[ii]))/np.sqrt(2*np.pi*vv(locs[ii]))
        output += d_area*pi_theta_quad[idx]*gauss_evals[:,:,idx]
    return output,gauss_evals
# %%
################# PROBLEM SETUP #################

## Import mesh

# number of degrees of freedom in mesh (selects which mesh to import)
# current options are 253, 399, 557, 605, 729, 1225, 1879, 2363, 2779, 5443, 8335, 11521
dofs = 729
mesh = dl.refine( dl.Mesh("meshes/adv_diff_dofs_{0}.xml".format(dofs)) )
Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
print( "Number of dofs: {0}".format( Vh.dim() ) )

# %%
## Set up prior and initial condition as a draw from the prior
gamma = 1.
delta = 8.
# covariance C = (delta * I - gamma * Laplacian)^{-2}
prior = BiLaplacianPrior(Vh, gamma, delta, robin_bc=True)
prior_mean = 0.
prior.mean = dl.interpolate(dl.Constant(prior_mean), Vh).vector()

noise = dl.Vector()
prior.init_vector(noise,"noise")
parRandom.normal(1., noise)
s_prior = dl.Function(Vh, name="sample_prior")
prior.sample(noise,s_prior.vector())
true_initial_condition = s_prior.vector()

# ic_expr = dl.Expression(
#     'std::min(0.5,std::exp(-100*(std::pow(x[0]-0.35,2) +  std::pow(x[1]-0.7,2))))',
#     element=Vh.ufl_element())
# true_initial_condition = dl.interpolate(ic_expr, Vh).vector()

# %%
## Plot initial condition and prior mean

objs = [dl.Function(Vh,true_initial_condition),
        dl.Function(Vh,prior.mean)]
mytitles = ["True Initial Condition", "Prior mean"]
nb.multi1_plot(objs, mytitles)
plt.show()

# %%
## Set up and solve forward problem
    
nt = 20
t_init         = 0.
t_final        = 4.
t_1            = 1.
dt             = t_final/nt
observation_dt = 0.4
simulation_times = np.arange(t_init, t_final+.5*dt, dt)
observation_times = np.arange(t_1, t_final+.5*dt, observation_dt)

# # targets along building edges
# targets = np.loadtxt('targets.txt')
pts = 4 # number of observation points on each side of the buildings
targets = np.zeros((8*pts,2))
targets[0:pts,0] = np.linspace(.25, .5, pts+2)[1:-1]
targets[0:pts,1] = 0.149
targets[pts:2*pts,0] = np.linspace(.25, .5, pts+2)[1:-1]
targets[pts:2*pts,1] = 0.401
targets[2*pts:3*pts,0] = 0.249
targets[2*pts:3*pts,1] = np.linspace(.15, .4, pts+2)[1:-1]
targets[3*pts:4*pts,0] = 0.501
targets[3*pts:4*pts,1] = np.linspace(.15, .4, pts+2)[1:-1]
targets[4*pts:5*pts,0] = np.linspace(.6, .75, pts+2)[1:-1]
targets[4*pts:5*pts,1] = 0.599
targets[5*pts:6*pts,0] = np.linspace(.6, .75, pts+2)[1:-1]
targets[5*pts:6*pts,1] = 0.851
targets[6*pts:7*pts,0] = 0.599
targets[6*pts:7*pts,1] = np.linspace(.6, .85, pts+2)[1:-1]
targets[7*pts:8*pts,0] = 0.751
targets[7*pts:8*pts,1] = np.linspace(.6, .85, pts+2)[1:-1]

# plot target locations
nb.plot(dl.Function(Vh,true_initial_condition),mytitle='Sensor Locations')
plt.scatter(targets[:,0],targets[:,1],color='red')

print ("Number of observation points: {0}".format(targets.shape[0]) )
print ("Number of observation times: {0}".format(observation_times.shape[0]) )
# initialize observations
misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, targets)

# initialize problem
wind_velocity = computeVelocityField(mesh)
kappa = 0.001
problem_true = TimeDependentAD(mesh, [Vh,Vh,Vh], prior, misfit, simulation_times, kappa, wind_velocity, True)

lam_true = 1e6  # true noise precision

# initialize vector in the state space
utrue = problem_true.generate_vector(STATE)
x = [utrue, true_initial_condition, None]
# solve forward problem
problem_true.solveFwd(x[STATE], x)
# observe solution and add error
misfit.observe(x, misfit.d)
noise_std_dev = 1/np.sqrt(lam_true)
parRandom.normal_perturb(noise_std_dev,misfit.d)
misfit.noise_variance = noise_std_dev**2

# plot solution
nb.show_solution(Vh, true_initial_condition, utrue, "Solution")
#plt.savefig("forward_solution.pdf",pad_inches=1)

# %%
## Define parameters

# note -- contains the true noise variance in misfit. Will be overwritten in computation.
model = Model(mesh, Vh, misfit, simulation_times, kappa, wind_velocity)

theta = np.array([gamma, delta, lam_true])

# hyperprior parameters (independent gamma distributions in gamma and delta, shifted to start or end at min/max values)
alpha_del = 1
alpha_gam = 1
alpha_lam = 1
beta_del = 1e-4
beta_gam = 1e-4
beta_lam = 1e-8
max_del = 50
min_gam = 0.2
max_lam = 1e7
hyp_pr_params = np.array([alpha_del, alpha_gam, alpha_lam, beta_del, beta_gam, beta_lam, max_del, min_gam, max_lam])
pretheta = [min_gam, max_del, max_lam]

#%%
# Plot errors as a function of rank to choose rank (for timing)

# "true" posterior variance
k = 200
lmbda_prior, V_prior = LowRankApprox([theta[0], theta[1], theta[2]], k, model)
posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda_prior, V_prior, [theta[0], theta[1], theta[2]], model)
posterior_var_true,pr,corr = posterior.pointwise_variance(method="Exact")
Mvar = dl.Vector(posterior.prior.R.mpi_comm())
posterior.prior.init_vector(Mvar,0)
posterior.prior.M.mult(posterior_var_true,Mvar)
var_norm_true = posterior_var_true.inner(Mvar)

# error in posterior variance for various ranks k
ks = np.arange(15, 100, 1)
threshold = 1e-3
errs_prior,k_p,lmbda_prior,V_prior = PostCovError(theta, [theta[0], theta[1], theta[2]], posterior_var_true, var_norm_true, ks, threshold, model)
errs_weak,k_w,lmbda_weak,V_weak = PostCovError(theta, pretheta, posterior_var_true, var_norm_true, ks, threshold, model)
errs_unprecon,k_u,lmbda_unprecon,V_unprecon = PostCovError(theta, [None, None, pretheta[2]], posterior_var_true, var_norm_true, ks, threshold, model)

# Plot spectra of low-rank approx for 3 methods
fig = plt.figure(figsize=(10,7.2))
plt.rcParams.update({'font.size': 16})
plt.semilogy(range(len(lmbda_unprecon)), lmbda_unprecon, label=f'unprecon, $\lambda$={max_lam:.1e}')
plt.semilogy(range(len(lmbda_weak)), lmbda_weak, label=fr'weakest, $\lambda$={max_lam:.1e}')
plt.semilogy(range(len(lmbda_prior)), lmbda_prior, label=f'prior precon, $\lambda$={lam_true:.1e}')
plt.title('Spectrum of Low Rank Approx')
plt.legend()

# Plot error as a function of rank for 3 methods
fig = plt.figure(figsize=(10,7.2))
plt.rcParams.update({'font.size': 16})
plt.semilogy(ks, errs_unprecon, label='unprecon')
plt.semilogy(ks, errs_weak, label='weakest precon')
plt.semilogy(ks, errs_prior, label='prior precon')
plt.xlabel('rank')
plt.ylabel('error')
plt.title('Relative error in pointwise posterior covariance')
plt.legend()

print(f'k_p = {k_p}, k_w = {k_w}, k_u = {k_u}')

#%% 
# One figure with multiple error plots
fig = plt.figure(figsize=(10,7.2))
plt.rcParams.update({'font.size': 16})
plt.semilogy(ks, errs_unprecon, color='green', ls='-', label=r'un, $\theta_1$')
plt.semilogy(ks, errs_weak, color='orange', ls='-', label=r'weak, $\theta_1$')
plt.semilogy(ks, errs_prior, color='blue', ls='-', label=r'prior, $\theta_1$')

theta = np.array([0.8,12,1.3e6])
k = 200
lmbda_prior, V_prior = LowRankApprox([theta[0], theta[1], theta[2]], k, model)
posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda_prior, V_prior, [theta[0], theta[1], theta[2]], model)
posterior_var_true,pr,corr = posterior.pointwise_variance(method="Exact")
Mvar = dl.Vector(posterior.prior.R.mpi_comm())
posterior.prior.init_vector(Mvar,0)
posterior.prior.M.mult(posterior_var_true,Mvar)
var_norm_true = posterior_var_true.inner(Mvar)
errs_prior,k_p,lmbda_prior,V_prior = PostCovError(theta, [theta[0], theta[1], theta[2]], posterior_var_true, var_norm_true, ks, threshold, model)
errs_weak,k_w,lmbda_weak,V_weak = PostCovError(theta, pretheta, posterior_var_true, var_norm_true, ks, threshold, model)
errs_unprecon,k_u,lmbda_unprecon,V_unprecon = PostCovError(theta, [None, None, pretheta[2]], posterior_var_true, var_norm_true, ks, threshold, model)
print(f'k_p = {k_p}, k_w = {k_w}, k_u = {k_u}')
plt.semilogy(ks, errs_unprecon, color='green', ls='--', label=r'un, $\theta_2$')
plt.semilogy(ks, errs_weak, color='orange', ls='--', label=r'weak, $\theta_2$')
plt.semilogy(ks, errs_prior, color='blue', ls='--', label=r'prior, $\theta_2$')

theta = np.array([1.4,3,7e5])
k = 200
lmbda_prior, V_prior = LowRankApprox([theta[0], theta[1], theta[2]], k, model)
posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda_prior, V_prior, [theta[0], theta[1], theta[2]], model)
posterior_var_true,pr,corr = posterior.pointwise_variance(method="Exact")
Mvar = dl.Vector(posterior.prior.R.mpi_comm())
posterior.prior.init_vector(Mvar,0)
posterior.prior.M.mult(posterior_var_true,Mvar)
var_norm_true = posterior_var_true.inner(Mvar)
errs_prior,k_p,lmbda_prior,V_prior = PostCovError(theta, [theta[0], theta[1], theta[2]], posterior_var_true, var_norm_true, ks, threshold, model)
errs_weak,k_w,lmbda_weak,V_weak = PostCovError(theta, pretheta, posterior_var_true, var_norm_true, ks, threshold, model)
errs_unprecon,k_u,lmbda_unprecon,V_unprecon = PostCovError(theta, [None, None, pretheta[2]], posterior_var_true, var_norm_true, ks, threshold, model)
print(f'k_p = {k_p}, k_w = {k_w}, k_u = {k_u}')
plt.semilogy(ks, errs_unprecon, color='green', ls=':', label=r'un, $\theta_3$')
plt.semilogy(ks, errs_weak, color='orange', ls=':', label=r'weak, $\theta_3$')
plt.semilogy(ks, errs_prior, color='blue', ls=':', label=r'prior, $\theta_3$')

plt.xlabel('rank')
plt.ylabel('error')
plt.title('Relative error in pointwise posterior covariance')
plt.legend()

# %%
# Find low rank approx of prior-to-posterior update

precon = 'unprecon'

if precon == 'unprecon':
    pretheta[0] = None; pretheta[1] = None
    k = k_u
elif precon == 'prior':
    k = k_p
else:
    assert precon == 'weakest', 'precon should have value prior, unprecon, or weakest'
    k = k_w

# %%
# Compute -log pi for a range of thetas
compute_start = time.time()

if precon == 'weakest' or precon == 'unprecon':
    lmbda, V = LowRankApprox(pretheta, k, model)

nn = 10
nl = 10
g_range = np.linspace(0.7,1.5,nn)
d_range = np.linspace(0.1,15,nn)
l_range = np.linspace(5e5, 1.5e6, nl)
logpi = np.zeros((len(g_range),len(d_range),len(l_range)))
print('Progress in indices computed from (0,0,0) to ({0},{0},{1}):'.format(nn-1,nl-1))
for i in range(len(g_range)):
    for j in range(len(d_range)):
        for k in range(len(l_range)):
            #compute -log pi_hat(theta)
            theta = np.array([g_range[i],d_range[j],l_range[k]])
            if precon == 'prior':
                pretheta = theta.tolist()
                lmbda, V = LowRankApprox(pretheta, k, model)
            logpi[i,j,k] = neglogpi_theta(theta, lmbda, V, pretheta, hyp_pr_params, model)
            print('({0},{1},{2})'.format(i,j,k))

compute_end = time.time()
print(f"Compute time: {compute_end-compute_start} seconds")

# %%

lam_idx = 3
plt.pcolormesh(d_range,g_range,logpi[:,:,lam_idx])
plt.set_cmap('bone')
plt.colorbar()
plt.title(fr'$-log \, \pi(\gamma, \delta, \lambda | y), \quad \lambda = {l_range[lam_idx]}$')
plt.ylabel(r'$\gamma$')
plt.xlabel(r'$\delta$')

# %%

# scaled arbitrarily to have max value 1 in order to avoid overflow errors
fig = plt.figure(figsize=(10,7.2))
plt.rcParams.update({'font.size': 16})
plt.set_cmap('bone')
plt.pcolormesh(d_range,g_range,np.exp(-logpi[:,:,lam_idx]+np.min(logpi[:,:,lam_idx])))
plt.colorbar()
# plt.title(r'$\pi(\gamma, \delta | y), dofs={0}$'.format(dofs))
plt.title(r'$\pi(\gamma, \delta, \lambda | y)$')
plt.ylabel(r'$\gamma$')
plt.xlabel(r'$\delta$')
# plt.savefig("pi_gamma_delta.pdf",bbox_inches='tight', pad_inches=0)

# %%
g_idx = 3; d_idx = 3
fig = plt.figure(figsize=(10,7.2))
plt.rcParams.update({'font.size': 16})
plt.plot(l_range,logpi[g_idx,d_idx,:])
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$-log \pi(\gamma, \delta, \lambda|y)$')

# %%
fig = plt.figure(figsize=(10,7.2))
plt.rcParams.update({'font.size': 16})
plt.plot(l_range,np.exp(-logpi[g_idx,d_idx,:]+np.min(logpi[g_idx,d_idx,:])))
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\pi(\gamma, \delta, \lambda|y)$')

 # %%

plt.pcolormesh(l_range,g_range,logpi[:,d_idx,:])
plt.set_cmap('bone')
plt.colorbar()
plt.title(fr'$-log \, \pi(\gamma, \delta, \lambda | y), \quad \delta = {d_range[d_idx]}$')
plt.ylabel(r'$\gamma$')
plt.xlabel(r'$\lambda$')

# %%

# scaled arbitrarily to have max value 1 in order to avoid overflow errors
fig = plt.figure(figsize=(10,7.2))
plt.rcParams.update({'font.size': 16})
plt.set_cmap('bone')
plt.pcolormesh(l_range,g_range,np.exp(-logpi[:,d_idx,:]+np.min(logpi[:,d_idx,:])))
plt.colorbar()
# plt.title(r'$\pi(\gamma, \delta | y), dofs={0}$'.format(dofs))
plt.title(rf'$\pi(\gamma, \delta, \lambda | y), \: \delta = {d_range[d_idx]:.2f}$')
plt.ylabel(r'$\gamma$')
plt.xlabel(r'$\lambda$')
# plt.savefig("pi_gamma_delta.pdf",bbox_inches='tight', pad_inches=0)

# %%
## Compute MAP point of pi(theta | y)

opt_start = time.time()
def neglogpi_helper(theta):
    return neglogpi_theta(theta, lmbda, V, pretheta, hyp_pr_params, model)
theta0 = np.array([1, 1, 1e6])
theta_opt = opt.minimize(neglogpi_helper,theta0,method='Nelder-Mead',options={'disp':True})
opt_end = time.time()
print(f"Optimization time: {opt_end-opt_start} seconds")

theta_MAP = theta_opt.x
print(f"MAP point of pi(theta|y): {theta_MAP}")

# %%
## Find inverse Hessian at MAP point

# choosing the finite difference dx's here is finicky -- can't be much smaller
dtheta = [1e-1,8e-1,1e3] # test last number options
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
def theta_of_z(z):
    return theta_MAP + np.dot(Hinv_V,np.dot(Hinv_L_sqrt,z))

# %%
# for each coordinate of z, find its values with significant probability
delta_z = 1
delta_pi = 2.5
maxiter = 20

z_highprob = [np.array([0.0]) for i in range(ntheta)]
for idx in range(ntheta):
    z = np.zeros(ntheta)
    z[idx] = delta_z
    count = 0
    while all(theta_of_z(z)>0) and neglogpi_helper(theta_of_z(z)) - neglogpi_helper(theta_MAP) < delta_pi and count < maxiter:
        z_highprob[idx] = np.append(z_highprob[idx],z[idx])
        z[idx] += delta_z
        count += 1
        print(count)
    count = 0
    z[idx] = -delta_z
    while all(theta_of_z(z)>0) and neglogpi_helper(theta_of_z(z)) - neglogpi_helper(theta_MAP) < delta_pi and count < maxiter:
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

# use to recursively find all combinations of possible points
all_points = [[zval] for zval in z_highprob[0]]
if ntheta > 1:
    for idx in range(1,ntheta):
        all_points = pt_pairs(all_points, z_highprob[idx])

# search through them for only the ones with high enough probability
# could be more efficient -- don't recalculate along axes, and/or store values for later
quad_points = []
for i in range(len(all_points)):
    theta_i = theta_of_z(np.array(all_points[i]))
    if neglogpi_helper(theta_i) - neglogpi_helper(theta_MAP) < delta_pi:
        quad_points.append(theta_i)
quad_points = np.array(quad_points)

# %%
# scatter plot of quadrature points (if 3D)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(quad_points[:,0],quad_points[:,1],quad_points[:,2],color='red',label='quadrature points') 
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\delta$')
ax.set_zlabel(r'$\lambda$')
ax.set_title('Quadrature Points')

## %%

# # plot scaled pi(gamma, delta | data) with quadrature points
# fig = plt.figure(figsize=(10,7.2))
# plt.rcParams.update({'font.size': 16})
# plt.set_cmap('bone')
# plt.pcolormesh(d_range,g_range,np.exp(-logpi+neglogpiMAP)*scale)
# plt.colorbar()
# plt.scatter(quad_points[:,1],quad_points[:,0],color='red',label='quadrature points') 
# plt.title(r'$\pi(\gamma, \delta | y)$')
# plt.ylabel(r'$\gamma$')
# plt.xlabel(r'$\delta$')
# plt.legend(loc='upper left') #, bbox_to_anchor=(0.9, 0.3))
# # plt.savefig("quad_points.pdf",bbox_inches='tight', pad_inches=0)

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

# # %%

# ### Upgrade to a finer mesh
# dofs = 2779
# mesh = dl.refine( dl.Mesh("adv_diff_dofs_{0}.xml".format(dofs)) )
# wind_velocity = computeVelocityField(mesh)
# Vh = dl.FunctionSpace(mesh, "Lagrange", 1)

# true_u0_fun = dl.Function(Vh, true_initial_condition) # this does not work
# # need to find a way to interpolate the initial condition from the earlier mesh
# true_initial_condition = dl.interpolate(true_u0_fun, Vh).vector()
# misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, targets)

# dt = dt/2
# simulation_times = np.arange(t_init, t_final+.5*dt, dt)

# dummy_prior = BiLaplacianPrior(Vh, 1., 8., robin_bc=True)
# dummy_prior.mean = dl.interpolate(dl.Constant(prior_mean), Vh).vector()

# if weakest_precon:
#     preprior = BiLaplacianPrior(Vh, pregamma, predelta, robin_bc=True)
#     preprior.mean = dl.interpolate(dl.Constant(prior_mean), Vh).vector()
#     problem_true = TimeDependentAD(mesh, [Vh,Vh,Vh], preprior, misfit, simulation_times, kappa_true, wind_velocity, True)
# else:
#     problem_true = TimeDependentAD(mesh, [Vh,Vh,Vh], dummy_prior, misfit, simulation_times, kappa_true, wind_velocity, True)

# # relative noise level
# rel_noise = 0.01
# # initialize vector in the state space
# utrue = problem_true.generate_vector(STATE)
# x = [utrue, true_initial_condition, None]
# # solve forward problem
# problem_true.solveFwd(x[STATE], x)
# # NEED TO FIX THIS! SHOULDN'T BE REPLACING THE DATA WHEN WE UPGRADE THE MESH
# # observe solution and add error
# misfit.observe(x, misfit.d)
# MAX = misfit.d.norm("linf", "linf")
# noise_std_dev = rel_noise * MAX
# parRandom.normal_perturb(noise_std_dev,misfit.d)
# misfit.noise_variance = noise_std_dev*noise_std_dev

# H_misfit_only = ReducedHessian(problem_true, misfit_only=True)
# k = 128
# pad = 20
# Omega = MultiVector(x[PARAMETER], k+pad)
# parRandom.normal(1., Omega)

# if weakest_precon:
#     lmbda, V = singlePassG(H_misfit_only, preprior.R, preprior.Rsolver, Omega, k) 
# else:
#     lmbda, V = singlePass(H_misfit_only, Omega, k)

# # overwriting the previous one, now for finer mesh
# def neglogpi_helper(theta):
#     logpi,det,pri,uQu,muQmu = neglogpi_gamma_delta(mesh, Vh, misfit, wind_velocity, lmbda, V, pregamma, predelta, kappa, theta[0], theta[1])
#     return logpi
# neglogpiMAP = neglogpi_helper(theta_MAP)

#%%
### Compute marginal distribution of QoI that is a linear scalar function of u_0
## QoI for now is the average of u_0 in part of the domain

# box location
xmin = 0.25; xmax = 0.5
ymin = 0.5; ymax = 0.75
boxlims = np.array([xmin, xmax, ymin, ymax])

# print QoI(constant 1 function) to test error introduced by finite element approx
testu0 = dl.interpolate(dl.Constant(1), Vh).vector()
print(f"QoI(constant 1 function) = {QoI(testu0, Vh, boxlims)}")

# %%

# evaluate pi(qoi|y) at range of qoi values
qoi_range = np.linspace(0.0,0.7,100)
pi_qoi = QoIdist(qoi_range, quad_points, pi_theta_quad, d_area, boxlims, lmbda, V, pretheta, model)
# might want to normalize again here -- this prob does not quite integrate to 1

#%%
theta_true = np.array([gamma, delta, lam_true])
theta_1 = np.array([1, 8, 1.15e6])
theta_2 = np.array([0.7, 8, 1e6])
pi_qoi_th_true = QoIdist_fixed_theta(qoi_range, theta_true, boxlims, lmbda, V, pretheta, model)
pi_qoi_th_1 = QoIdist_fixed_theta(qoi_range, theta_1, boxlims, lmbda, V, pretheta, model)
pi_qoi_th_2 = QoIdist_fixed_theta(qoi_range, theta_2, boxlims, lmbda, V, pretheta, model)

# true QoI
true_QoI = QoI(true_initial_condition, Vh, boxlims)

#%%
# plot distribution of QoI
plt.figure(figsize=(10,4.5))
plt.rcParams.update({'font.size': 16})
plt.plot(qoi_range,pi_qoi,linewidth=3,color='black', label=r"marginalized")
plt.plot(qoi_range,pi_qoi_th_true,linewidth=3,color='green', label=rf"true $\theta = [{theta_true[0]}, {theta_true[1]}, {theta_true[2]:.0e}]$")
plt.plot(qoi_range,pi_qoi_th_1,linewidth=3,color='red', label=rf"$\theta = [{theta_1[0]}, {theta_1[1]}, {theta_1[2]:.2e}]$")
plt.plot(qoi_range,pi_qoi_th_2,linewidth=3,color='orange', label=rf"$\theta = [{theta_2[0]}, {theta_2[1]}, {theta_2[2]:.0e}]$")
plt.axvline(x=true_QoI, color='black', linestyle="-.", label=r"true qoi")
plt.title(f'Posterior Distribution of QoI')
plt.ylabel(r"$\pi(qoi|y)$")
plt.xlabel(r"qoi")
plt.tight_layout()
plt.legend()

# plot -log distribution of QoI 
plt.figure(figsize=(10,4.5))
plt.rcParams.update({'font.size': 16})
plt.plot(qoi_range,-np.log(pi_qoi),linewidth=3,color='black', label=r"marginalized")
plt.plot(qoi_range,-np.log(pi_qoi_th_true),linewidth=3,color='green', label=rf"true $\theta = [{theta_true[0]}, {theta_true[1]}, {theta_true[2]:.0e}]$")
plt.plot(qoi_range,-np.log(pi_qoi_th_1),linewidth=3,color='red', label=rf"$\theta = [{theta_1[0]}, {theta_1[1]}, {theta_1[2]:.2e}]$")
plt.plot(qoi_range,-np.log(pi_qoi_th_2),linewidth=3,color='orange', label=rf"$\theta = [{theta_2[0]}, {theta_2[1]}, {theta_2[2]:.0e}]$")
plt.axvline(x=true_QoI, color='black', linestyle="-.", label=r"true qoi")
plt.title(f'Posterior Distribution of QoI')
plt.ylabel(r"$-log \pi(qoi|y)$")
plt.xlabel(r"qoi")
plt.tight_layout()
plt.legend(loc="upper right")

# %%
u_range = np.linspace(0.0,0.55,100)
locations = [[0.3,0.7],[0.45,0.55]]
true_u0_fun = dl.Function(Vh,true_initial_condition)
pi_u_0_i_evals,gauss_evals = posterior_marginals(locations, u_range, quad_points, pi_theta_quad, d_area, lmbda, V, pretheta, model)
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
