import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .inverse_problem import LowRankApprox, ComputePosterior, multivector_slice

def make_single_spectrum_plot(pretheta, lmbda):
    """ plots eigenvalues, scaled to coincide with the definition in the paper """
    fig = plt.figure(figsize=(10,7.2))
    plt.rcParams.update({'font.size': 20})
    plt.semilogy(range(len(lmbda)), lmbda*(pretheta[2]**2)*(pretheta[1]**2), linewidth=3)
    plt.ylabel(r'$\lambda_i$')
    plt.xlabel(r'$i$')
    plt.show()

def compute_all_spectra(thetas, rank, low_rank_tolerance, hyp_pr_params, problem, theta_ref=None, *, save_scaled=False):
    """ Compute UP, WP, and several PP spectra"""
    min_eta = hyp_pr_params["min_eta"]
    min_delta = hyp_pr_params["min_delta"]
    min_sigma = hyp_pr_params["min_sigma"]
    
    lmbda_priors = [0 for idx in thetas]
    V_priors = [0 for idx in thetas]
    for idx in range(len(thetas)):
        lmbda_priors[idx], V_priors[idx] = LowRankApprox(thetas[idx], rank, problem)
    lmbda_weak, V_weak = LowRankApprox(np.array([min_eta, 1.0, 1.0]), rank, problem)
    lmbda_unprecon, V_unprecon = LowRankApprox(np.array([0.0, 1.0, 1.0]), rank, problem)

    # Ranks for min cutoff
    min_cutoff = low_rank_tolerance * min_sigma**2 * min_delta**2
    r_w_min = np.argmax(lmbda_weak < min_cutoff)
    r_u_min = np.argmax(lmbda_unprecon < min_cutoff)
    print(f'min cutoff ranks: weak = {r_w_min}, unprecon = {r_u_min}')

    # Ranks for updated cutoff if given
    if theta_ref is not None:
        cutoff_ref = low_rank_tolerance * theta_ref[1]**2 * theta_ref[2]**2
        r_w_ref = np.argmax(lmbda_weak < cutoff_ref)
        r_u_ref = np.argmax(lmbda_unprecon < cutoff_ref)
        print(f'reference theta cutoff ranks: weak = {r_w_ref}, unprecon = {r_u_ref}')

    if save_scaled:
        # scale by sigma^2 delta^2 for consistency with definition of eigenvalues in the paper
        filename = "images/spectra_full.txt"
        header = "r \t\t unprecon \t\t weakest"
        spectra = np.column_stack((np.arange(1,rank+1,1), lmbda_unprecon, lmbda_weak))
        for idx in range(len(thetas)):
            header = header + f" \t\t prior{idx+1}"
            spectra = np.column_stack((spectra, lmbda_priors[idx]*(thetas[idx][2]**2)*(thetas[idx][1]**2)))
        np.savetxt(filename, spectra, delimiter="\t", header=header, fmt='%10.14f', comments="")

    return (lmbda_unprecon, lmbda_weak, lmbda_priors, V_unprecon, V_weak, V_priors)

def plot_all_spectra(thetas, lmbda_weak, lmbda_unprecon, lmbda_priors):
    """ Plot UP, WP, and several PP spectra """
    # scale by sigma^2 delta^2 for consistency with definition of eigenvalues in the paper
    lmbda_priors_scaled = [np.zeros(1) for idx in thetas]
    for idx in range(len(thetas)):
        lmbda_priors_scaled[idx] = lmbda_priors[idx]*(thetas[idx][2]**2)*(thetas[idx][1]**2)

    fig = plt.figure(figsize=(10,7.2))
    plt.rcParams.update({'font.size': 20})
    plt.semilogy(range(len(lmbda_unprecon)), lmbda_unprecon, linewidth=3, label='UP')
    plt.semilogy(range(len(lmbda_weak)), lmbda_weak, linewidth=3, label='WP')
    for idx in range(len(thetas)):
        plt.semilogy(range(len(lmbda_priors_scaled[idx])), lmbda_priors_scaled[idx], linewidth=3, label=rf'PP, $\theta_{idx}$')
    plt.ylim(bottom=1e-16)
    plt.ylabel(r'$\Lambda_{ii}$')
    plt.xlabel(r'$i$')
    plt.legend()


def compute_error_vs_rank(ranks, theta, eigendecompositions, neg_adj_y, hyp_pr_params, problem, cutoff=1e-2, *, save=False):
    lmbda_un, lmbda_weak, lmbda_pr, V_un, V_weak, V_pr = eigendecompositions
    posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda_pr, V_pr, neg_adj_y, theta, problem, use_CG=True)
    uQu = 0.5*mg.inner(posterior.mean)
    det_ratio = 0.0
    for ll in posterior.d:
        det_ratio += 0.5*np.log(1+ll)

    e1_pr = np.zeros(len(ranks)); e2_pr = np.zeros(len(ranks))
    e1_weak = np.zeros(len(ranks)); e2_weak = np.zeros(len(ranks))
    e1_un = np.zeros(len(ranks)); e2_un = np.zeros(len(ranks))
    for idx in range(len(ranks)):
        r_trunc = int(ranks[idx])
        lmbda_trunc_pr = lmbda_pr[0:r_trunc-1]
        lmbda_trunc_weak = lmbda_weak[0:r_trunc-1]
        lmbda_trunc_un = lmbda_un[0:r_trunc-1]
        V_trunc_pr = multivector_slice(V_pr, r_trunc-1)
        V_trunc_weak = multivector_slice(V_weak, r_trunc-1)
        V_trunc_un = multivector_slice(V_un, r_trunc-1)

        posteriorNoCG_pr,mgNoCG_pr,lmbda_new,V_new = ComputePosterior(theta, lmbda_trunc_pr, 
                V_trunc_pr, neg_adj_y, theta, problem)
        e1_pr[idx] = det_ratio
        for ll in posteriorNoCG_pr.d:
            e1_pr[idx] -= 0.5*np.log(1+ll)
        e2_pr[idx] = uQu-0.5*mgNoCG_pr.inner(posteriorNoCG_pr.mean)

        posteriorNoCG_weak,mgNoCG_weak,lmbda_new,V_new = ComputePosterior(theta, lmbda_trunc_weak, 
                V_trunc_weak, neg_adj_y, np.array([hyp_pr_params['min_eta'], 1.0, 1.0]), problem)
        e1_weak[idx] = det_ratio
        for ll in posteriorNoCG_weak.d:
            e1_weak[idx] -= 0.5*np.log(1+ll)
        e2_weak[idx] = uQu-0.5*mgNoCG_weak.inner(posteriorNoCG_weak.mean)

        posteriorNoCG_un,mgNoCG_un,lmbda_new,V_new = ComputePosterior(theta, lmbda_trunc_un, 
                V_trunc_un, neg_adj_y, np.array([0,1,1]), problem)
        e1_un[idx] = det_ratio
        for ll in posteriorNoCG_un.d:
            e1_un[idx] -= 0.5*np.log(1+ll)
        e2_un[idx] = uQu-0.5*mgNoCG_un.inner(posteriorNoCG_un.mean)

    r_p_err = int(ranks[0]) + np.argmax((e1_pr+e2_pr) < cutoff)+1
    r_w_err = int(ranks[0]) + np.argmax((e1_weak+e2_weak) < cutoff)+1
    r_u_err = int(ranks[0]) + np.argmax((e1_un+e2_un) < cutoff)+1
    print(f"PP, WP, and UP ranks to reach total error {cutoff}: {r_p_err}, {r_w_err}, {r_u_err}")

    if save:
        filename = "images/log_pi_error.txt"
        header = "r \t\t e1_pr \t\t e1_weak \t\t e1_un \t\t e2_pr \t\t e2_weak \t\t e2_un"
        data = np.column_stack((ranks, e1_pr, e1_weak, e1_un, e2_pr, e2_weak, e2_un))
        np.savetxt(filename, data, delimiter="\t", header=header, fmt='%10.14f', comments="")

    return (e1_pr, e1_weak, e1_un, e2_pr, e2_weak, e2_un)


def plot_error_vs_rank(ranks, errors):
    e1_pr, e1_weak, e1_un, e2_pr, e2_weak, e2_un = errors
    plt.semilogy(ranks, e1_pr, color="green", label="e1 PP")
    plt.semilogy(ranks, e2_pr, color="green", linestyle="--", label="e2 PP")
    plt.semilogy(ranks, e1_weak, color="orange", label="e1 WP")
    plt.semilogy(ranks, e2_weak, color="orange", linestyle="--", label="e2 WP")
    plt.semilogy(ranks, e1_un, color="blue", label="e1 UP")
    plt.semilogy(ranks, e2_un, color="blue", linestyle="--", label="e2 UP")
    plt.legend()
    plt.xlabel('rank')
    plt.ylabel('error')
    plt.show()

def plot_quad_points(quad_points, thetas=None, theta_MAP=None, *, save=False):
    plt.rcParams.update({
        "font.family": "serif",       
        "mathtext.fontset": "cm",
        "font.size": 18,
        "figure.dpi": 300             
    })

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection='3d')
    ax.computed_zorder = False

    # white background
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # light gray grid lines
    grid_style = {'color': (0.9, 0.9, 0.9, 1.0), 'linewidth': 0.5}
    ax.xaxis._axinfo["grid"].update(grid_style)
    ax.yaxis._axinfo["grid"].update(grid_style)
    ax.zaxis._axinfo["grid"].update(grid_style)

    # black quadrature points
    ax.scatter(quad_points[:, 0], quad_points[:, 2], quad_points[:, 1], 
            color='black', s=20, edgecolors='none', depthshade=True, zorder=1, label='quad. pts.')
    # three green points for reference thetas
    if thetas is not None:
        green_shades = ['#b3e600', '#00e600', '#009900'] 
        ax.scatter(thetas[0][0], thetas[0][2], thetas[0][1], color=green_shades[0], s=95, 
                   edgecolors='none', depthshade=True, label=r'$\theta_1$') 
        ax.scatter(thetas[1][0], thetas[1][2], thetas[1][1], color=green_shades[1], s=95, 
                   edgecolors='none', depthshade=True, label=r'$\theta_2$') 
        ax.scatter(thetas[2][0], thetas[2][2], thetas[2][1], color=green_shades[2], s=95, 
                   edgecolors='none', depthshade=True, label=r'$\theta_3$')

    # red MAP point 
    ax.scatter(theta_MAP[0], theta_MAP[2], theta_MAP[1], 
            color='red', s=190, edgecolors='none', depthshade=False, alpha=1.0, zorder=3, 
            label=r'$\theta_{\mathrm{MAP}}$') 

    # axis labels
    ax.set_xlabel(r'$\gamma$', fontsize=22, labelpad=14)
    ax.set_ylabel(r'$\sigma$', fontsize=22, labelpad=30)
    # ax.set_zlabel(r'$\delta$', fontsize=22, labelpad=30)
    ax.text2D(1.06, 0.6, r'$\delta$', transform=ax.transAxes, fontsize=22, 
            va='center', ha='center', rotation=0)

    # limit the number of ticks
    ax.xaxis.get_major_locator().set_params(nbins=5)
    ax.yaxis.get_major_locator().set_params(nbins=5)
    ax.zaxis.get_major_locator().set_params(nbins=5)

    ax.xaxis.set_tick_params(pad=4)
    ax.yaxis.set_tick_params(pad=0)
    ax.zaxis.set_tick_params(pad=8)

    # force the sigma (Y) tick text alignment to the right/bottom-right
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('left')
        tick.label1.set_verticalalignment('top')

    # ax.legend(loc='upper right', bbox_to_anchor=(0.3, 0.95), frameon=True, 
    #           facecolor='white', edgecolor='none', framealpha=0.9)

    fig.subplots_adjust(left=0.05, right=0.83, bottom=0.05, top=0.95)

    if save:
        fig.savefig("images/quad_points.pdf")


def save_experiment_results(
    output_dir,
    lmbda,
    theta_map,
    quad_points,
    pi_theta_quad,
    pi_qoi,
    pi_qoi_theta_map,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir / "lmbda.npy",
        lmbda,
    )
    np.save(
        output_dir / "theta_map.npy",
        theta_map,
    )
    np.save(
        output_dir / "quad_points.npy",
        quad_points,
    )
    np.save(
        output_dir / "pi_theta_quad.npy",
        pi_theta_quad,
    )
    np.save(
        output_dir / "pi_qoi.npy",
        pi_qoi,
    )
    np.save(
        output_dir / "pi_qoi_theta_map.npy",
        pi_qoi_theta_map,
    )
