#%%
import dolfin as dl
import numpy as np

from experiment_config import TimeConfig, ForwardConfig
from hyperparam_marginal import setup_adv_diff, solve_fwd_problem, make_neg_log_pi_theta, evaluate_pi_theta_on_grid, argmax_theta
from quadrature import find_quad_points, uniform_hyperprior_support
from box_average_qoi import compute_all_qoi_distributions
from plotting import compute_all_spectra, plot_all_spectra, compute_error_vs_rank, plot_error_vs_rank, plot_quad_points, save_experiment_results

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
dl.set_log_active(False)

#%%

# =============================================================================
# Configuration parameters
# =============================================================================

# Problem dimension
dim = 2

# Random seed
random_seed = 42

# Time discretization
time_config = TimeConfig(
    nt=80,
    t_init=0.0,
    t_final=4.0,
    first_observation_time=2.4,
    observation_dt=0.4,
)

# Data-generation noise
sigma_true = 1e-2

# Prior mean
prior_mean = 0.0

# Low-rank approximation
rank_mode = "fixed"
calibration_rank = 250
low_rank_tolerance = 1e-2
preconditioner = "weakest"

use_CG = False

# Optimization
theta_init = np.array([1, 10, 0.05])
optimization_method = 'Nelder-Mead'
optim_options = {'xatol':1e-2,'fatol':1e-2}
optim_verbose = True

# Quadrature and QoI
quadrature_search_maxiter = 20
num_qoi_values = 100

# =============================================================================
# Dimension-dependent configuration parameters
# =============================================================================

if dim == 2:
    mesh_vertices = 2363
    forward_config = ForwardConfig(
        dim=dim,
        mesh_vertices=mesh_vertices,
        mesh_path=(f"meshes/adv_diff_dofs_{mesh_vertices}.xml"),
        target_path="targets/targets_2d.txt",
        kappa=0.001,
        sigma_true=sigma_true,
        prior_mean=prior_mean,
        time=time_config)

    # hyperprior parameters (independent, uniform in [min,max])
    hyp_pr_params = {
        'min_eta': 0.0015,
        'max_eta': 10.0,
        'min_delta': 1.0,
        'max_delta': 100.0,
        'min_sigma': 3e-3,
        'max_sigma': 1e-1}

    reference_thetas = [
        np.array([0.003, 50, 0.01]),
        np.array([0.0075, 25, 0.01]),
        np.array([0.015, 12.5, 0.01])]
    qoi_theta_idxs = [0,2] # use 1st and 3rd reference thetas in qoi plot

    calibrated_ranks = {
        "prior": 50,
        "weakest": 95,
        "unpreconditioned": 110}

    evaluation_range = {
        "eta": (0.0015, 0.02),
        "delta": (15.0, 60.0),
        "sigma": (8.5e-3, 1.2e-2)}

    quadrature_dtheta = np.array([1e-3, 8e-1, 1e-5])
    quadrature_delta_z = 1.0
    quadrature_delta_pi = 2.5

    qoi_box = np.array([
        0.2, 0.4,
        0.7, 0.9,
    ])

    qoi_range_bounds = (0.1, 0.275)

elif dim == 3:
    mesh_vertices=7480
    forward_config = ForwardConfig(
        dim=dim,
        mesh_vertices=mesh_vertices,
        mesh_path=(f"velocity_fields/velocity_field_{mesh_vertices}.h5"),
        target_path="targets/targets_3d.txt",
        kappa=0.003,
        sigma_true=sigma_true,
        prior_mean=prior_mean,
        time=time_config)

    hyp_pr_params = {
        'min_eta': 0.01,
        'max_eta': 10.0,
        'min_delta': 1.0,
        'max_delta': 100.0,
        'min_sigma': 3e-3,
        'max_sigma': 1e-1}

    reference_thetas = [
        np.array([0.02, 60, 0.01]),
        np.array([0.04, 30, 0.01]),
        np.array([0.08, 15, 0.01])]
    qoi_theta_idxs = [0,2] # use 1st and 3rd reference thetas in qoi plot

    calibrated_ranks = {
        "prior": 100,
        "weakest": 199,
        "unpreconditioned": 237}

    evaluation_range = {
        "eta": (0.01, 0.2),
        "delta": (2.0, 80.0),
        "sigma": (1e-2, 1e-2)}

    quadrature_dtheta = np.array([1e-3, 2.0, 1e-5])
    quadrature_delta_z = 0.8
    quadrature_delta_pi = 3.0

    qoi_box = np.array([
        0.15, 0.3,
        0.7, 0.85,
        0.5, 0.65,
    ])

    qoi_range_bounds = (0.05, 0.25)

else:
    raise ValueError("dim must be either 2 or 3")

#%% 
# =============================================================================
# Setup
# =============================================================================

np.random.seed(random_seed)

problem_setup = setup_adv_diff(forward_config, velocity_plot=True)

problem, neg_adj_y = solve_fwd_problem(problem_setup, forward_config, plot=True, save=False)

#%%
# =============================================================================
# (Optional) Spectra and Error vs. Rank Plots
# =============================================================================

make_full_spectra_plot = False
make_rank_error_plot = False
# theta = theta_3 in error plot
idx_error_plot_theta = 2 
# theta = theta_3 used for example online ranks (Table 5.1 & 5.2, row 4)
reference_theta_for_cutoff = reference_thetas[2]
err_cutoff = 1e-2

# Figure 5.3, left (2D) and 5.6, left (3D): Low rank spectra
# Table 5.1 & 5.2, rows 3 and 4: ranks from cutoffs
if make_full_spectra_plot or make_rank_error_plot:
    eigendecomp_all_thetas = compute_all_spectra(reference_thetas, calibration_rank, 
            low_rank_tolerance, hyp_pr_params, problem, reference_theta_for_cutoff, save=False)
    lmbda_unprecon, lmbda_weak, lmbda_priors, V_unprecon, V_weak, V_priors = eigendecomp_all_thetas
if make_full_spectra_plot: 
    plot_all_spectra(reference_thetas, lmbda_weak, lmbda_unprecon, lmbda_priors)

# Figure 5.3, right: Error vs rank
# Table 5.1, row 5: ranks from error
if make_rank_error_plot:
    ranks_for_error_plot = np.linspace(11, 80, 70)
    eigendecomp_ref_theta = (lmbda_unprecon, lmbda_weak, lmbda_priors[idx_error_plot_theta], 
            V_unprecon, V_weak, V_priors[idx_error_plot_theta])
    errors = compute_error_vs_rank(ranks_for_error_plot, reference_thetas[idx_error_plot_theta], 
            eigendecomp_ref_theta, neg_adj_y, hyp_pr_params, problem, err_cutoff, save=False)
    plot_error_vs_rank(ranks_for_error_plot, errors)

#%% 
# =============================================================================
# Precompute Low-Rank Approximation (for WP, UP)
# =============================================================================

# Sets up -log pi(theta) function. Computes low rank approx if WP or UP.
low_rank = make_neg_log_pi_theta(preconditioner, rank_mode, 
        calibration_rank, calibrated_ranks, reference_thetas, low_rank_tolerance, neg_adj_y, 
        hyp_pr_params, problem, use_CG)
neg_log_hyperparam_marginal = low_rank.objective

# =============================================================================
# (Optional) Evaluate Hyperparameter Marginal on a Grid
# =============================================================================

# Figure 5.4, left (contours): Evaluate pi(theta) on a grid
# Tables 5.1 & 5.2, rows 1 & 2: timings computed using %%prun on this cell
grid_evaluate = False
grid_resolution = {
    "eta": 1,
    "delta": 1,
    "sigma": 1}
if grid_evaluate:
    evaluate_pi_theta_on_grid(evaluation_range, grid_resolution, neg_log_hyperparam_marginal, 
            plot=False, save=False)

# %% 
# =============================================================================
# Quadrature
# =============================================================================

# Compute MAP point of pi(theta | y)
theta_MAP = argmax_theta(neg_log_hyperparam_marginal, theta_init, optimization_method, optim_options, optim_verbose)

# Find quadrature points
hyperprior_support = lambda theta: uniform_hyperprior_support(theta, hyp_pr_params)
quad_points, pi_theta_quad, d_area = find_quad_points(neg_log_hyperparam_marginal, theta_MAP, 
        quadrature_dtheta, quadrature_delta_z, quadrature_delta_pi, quadrature_search_maxiter, 
        hyperprior_support)

# Figure 5.6, right: Scatter plot of quadrature points
make_quad_point_plot = True
if make_quad_point_plot:
    plot_quad_points(quad_points, reference_thetas, theta_MAP, save=False)

#%% 
# =============================================================================
# Quantity of Interest
# =============================================================================

# Fig 5.4, right: Distributions of QoI
# comparison thetas, default theta_1 and theta_3
qoi_thetas = [reference_thetas[idx] for idx in qoi_theta_idxs] 

if preconditioner == "weakest" or preconditioner == "unpreconditioned":
    pi_qoi, pi_qoi_th_MAP, pi_qoi_thetas, true_QoI = compute_all_qoi_distributions(
                theta_MAP, qoi_thetas, qoi_box, quad_points, pi_theta_quad, d_area, 
                qoi_range_bounds, num_qoi_values, low_rank, dim, problem_setup.Vh, problem_setup.Vh2, 
                problem_setup.true_initial_condition, neg_adj_y, 
                problem, use_CG, qoi_theta_idxs, plot=True, save=True)
else:
    raise NotImplementedError("Computing QoI distributions is not implemented for prior preconditioning.")


# %%
# =============================================================================
# Save partial results for testing
# =============================================================================

save_results = True
if save_results:
    output_dir = (f"results/{dim}d_{preconditioner}")
    save_experiment_results(output_dir, low_rank.eigenvalues, theta_MAP, quad_points, pi_theta_quad,
        pi_qoi, pi_qoi_th_MAP)
# %%
