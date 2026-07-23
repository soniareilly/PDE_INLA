import dolfin as dl
import numpy as np
from hippylib import *
from hippylib_changes import *
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT
from hippylib.modeling.reducedHessian import ReducedHessian
from hippylib.algorithms.lowRankOperator import LowRankOperator
from hippylib.modeling.posterior import GaussianLRPosterior

def ComputePosterior(theta, lmbda, V, neg_adj_y, pretheta, problem, use_CG=False):
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
    
    prior = BiLaplacianPrior(problem.Vh[PARAMETER], eta*delta, delta, robin_bc=True)
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
        preprior = BiLaplacianPrior(problem.Vh[PARAMETER], preeta*predelta, predelta, robin_bc=True)
        preprior.mean = dl.interpolate(dl.Constant(0.), problem.Vh[PARAMETER]).vector()
        # replace preprior with prior
        W = MultiVector(V)
        for i in range(V.nvec()):
            preprior.R.mult(V[i], W[i])
        H_temp = LowRankOperator(lmbda, W)
        k = V.nvec()
        pad = 20 
        Omega = MultiVector(problem.generate_vector(PARAMETER), k+pad)
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

    preprior = BiLaplacianPrior(problem.Vh[PARAMETER], preeta*predelta, predelta, robin_bc=True)
    preprior.mean = dl.interpolate(dl.Constant(0.), problem.Vh[PARAMETER]).vector()
    problem.prior = preprior
    problem.misfit.noise_variance = presigma**2
        
    H_misfit_only = ReducedHessian(problem, misfit_only=True)
    pad = int(k/2)
    Omega = MultiVector(problem.generate_vector(PARAMETER), k+pad)
    parRandom.normal(1., Omega)

    lmbda, V = singlePassG(H_misfit_only, preprior.R, preprior.Rsolver, Omega, k) 
    return lmbda, V

# Helper function for slicing multivectors
def mv_k(mv, n):
    mv_n = MultiVector(mv[0], n)
    for i in range(n):
        mv_n[i].zero()
        mv_n[i].axpy(1.0, mv[i])
    return mv_n

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