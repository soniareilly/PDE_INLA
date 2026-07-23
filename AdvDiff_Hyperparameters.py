 # %%
import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.optimize as opt
from hippylib import *
from hippylib.modeling.variables import STATE, PARAMETER, ADJOINT

from hippylib_changes import *
from hyperparam_marginal import *
from quadrature import *
from box_average_qoi import *

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
dl.set_log_active(False)

import time
np.random.seed(42)

# %%
dim = 2
# ******************** 2D Problem Setup ****************************
if dim == 2:
    ## Import 2D mesh
    # number of vertices in mesh (selects which mesh to import)
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
    dl.plot(vh)
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
    show_solution(Vh2, ic_Vh2, utrue, mytitle="Solution")
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
precon = 'weakest'

if precon == 'unprecon':
    pretheta[0] = 0.0
    r = r_u
elif precon == 'prior':
    r = r_p
else:
    assert precon == 'weakest', 'precon should have value prior, unprecon, or weakest'
    r = r_w

# %% Compute -log pi for a range of thetas
if precon == 'weakest' or precon == 'unprecon':
    lmbda, V = LowRankApprox(pretheta, r, problem)

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

# scaled to have max value 1 in order to avoid overflow errors
pitheta = np.exp(-logpi+np.min(logpi))

save_pi_theta = True
if save_pi_theta:
    etamesh,dmesh,smesh = np.meshgrid(eta_range, d_range, s_range, indexing='ij')
    header = "eta \t\t delta \t\t sigma \t\t pi_theta"
    np.savetxt(f"images/pi_theta_{ne}x{nd}x{ns}.txt", np.column_stack((etamesh.ravel(), dmesh.ravel(), smesh.ravel(), pitheta.ravel())), delimiter="\t", header=header, fmt=('%g', '%g', '%g', '%e'), comments="")

# plot pi_theta as a function of eta and delta
if ne > 2 and nd > 2:
    sig_idx = int(ns/2)
    fig = plt.figure(figsize=(10,7.2))
    plt.rcParams.update({'font.size': 16})
    plt.set_cmap('bone')
    plt.pcolormesh(d_range,eta_range,pitheta[:,:,sig_idx])
    plt.colorbar()
    plt.title(r'$\pi(\eta, \delta, \sigma | y)$')
    plt.ylabel(r'$\eta$')
    plt.xlabel(r'$\delta$')

# plot pi_theta as a function of sigma
if ns > 2:
    eta_idx = int(ne/2); d_idx = int(nd/2)
    fig = plt.figure(figsize=(10,7.2))
    plt.rcParams.update({'font.size': 16})
    plt.plot(s_range,pitheta[eta_idx,d_idx,:])
    plt.xlabel(r'$\sigma$')
    plt.ylabel(r'$\pi(\eta, \delta, \sigma|y)$')

# %%
## Compute MAP point of pi(theta | y)

opt_start = time.time()
def neglogpi_helper(theta):
    return neglogpi_theta(theta, lmbda, V, tol, neg_adj_y, pretheta, hyp_pr_params, problem)
def opt_callback(intermediate_result):
    print(f"Current x: {intermediate_result.x}")
    print(f"Objective value: {intermediate_result.fun}")
theta0 = np.array([1, 10, 0.05])
theta_opt = opt.minimize(neglogpi_helper,theta0,method='Nelder-Mead',callback=opt_callback, options={'disp':True,'xatol':1e-2,'fatol':1e-2})
opt_end = time.time()
print(f"Optimization time: {opt_end-opt_start} seconds")

theta_MAP = theta_opt.x
print(f"MAP point of pi(theta|y): {theta_MAP}")

# %% Find quadrature points

# choosing the finite difference deltas here is finicky -- can't be much smaller
if dim == 2:
    dtheta = [1e-3,8e-1,1e-5]
    delta_z = 1
    delta_pi = 2.5
elif dim == 3:
    dtheta = [1e-3,2,1e-5]
    delta_z = 0.8
    delta_pi = 3
maxiter = 20

# tests if theta is in the support of the hyperprior, to avoid quad points outside the support
def hyperprior_support(theta):
    return uniform_hyperprior_support(theta, hyp_pr_params)
# find quadrature points and their pi_theta values
quad_points, pi_theta_quad, d_area = find_quad_points(neglogpi_helper, theta_MAP, dtheta, delta_z, delta_pi, maxiter, hyperprior_support)

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


#%%
### Compute marginal distribution of QoI that is a linear scalar function of u_0
## QoI here is the average of u_0 in part of the domain

# box location
if dim == 2:
    xmin = 0.2; xmax = 0.4
    ymin = 0.7; ymax = 0.9
    boxlims = np.array([xmin, xmax, ymin, ymax])
    fig, ax = plt.subplots()
    ic = dl.Function(Vh)
    ic.vector()[:] = true_initial_condition
    plt.sca(ax)
    plot_obj = dl.plot(ic)
    ax.set_title("IC and Box Location")    
    rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    fig.colorbar(plot_obj, ax=ax)
elif dim == 3:
    xmin = 0.15; xmax = 0.3
    ymin = 0.7; ymax = 0.85
    zmin = 0.5; zmax = 0.65
    boxlims = np.array([xmin, xmax, ymin, ymax, zmin, zmax])


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

save_qoi_data = False
if save_qoi_data:
    header = "q \t\t theta_opt \t\t theta_1 \t\t theta_3 \t\t marginalized"
    np.savetxt("images/piQoI.txt", np.column_stack((qoi_range, pi_qoi_th_true, pi_qoi_th_1, pi_qoi_th_3, pi_qoi)), delimiter="\t", header=header, fmt='%10.8f', comments="")


# %%
