import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from hippylib import *
from hippylib.algorithms.cgsolverSteihaug import CGSolverSteihaug
from . import hippylib_extensions as hc
from .inverse_problem import ComputePosterior

# average function of a box in the domain
# handles 2d and 3d
class BoxAverage(dl.UserExpression):
    def __init__(self, boxlims, **kwargs):
        super().__init__(**kwargs)
        self.boxlims = boxlims
        self.xmin = boxlims[0]; self.xmax = boxlims[1]
        self.ymin = boxlims[2]; self.ymax = boxlims[3]
        if boxlims.size == 6:
            self.zmin = boxlims[4]; self.zmax = boxlims[5]
    def eval(self, value, x):
        if self.boxlims.size == 4 and self.xmin < x[0] < self.xmax and self.ymin < x[1] < self.ymax:
            value[0] = 1.0/(self.xmax-self.xmin)/(self.ymax-self.ymin)
        elif self.boxlims.size == 6 and self.xmin < x[0] < self.xmax and self.ymin < x[1] < self.ymax and self.zmin < x[2] < self.zmax:
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
def QoIdist_fixed_theta(qoi, theta, boxlims, lmbda, V, neg_adj_y, pretheta, problem, use_CG=False, omega_full=None):
    output = np.zeros(len(qoi))
    # find Gaussian pi(qoi|theta,y)
    prior = hc.BiLaplacianPrior(problem.Vh[PARAMETER], theta[0]*theta[1], theta[1], robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(0.), problem.Vh[PARAMETER]).vector()
    posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda, V, neg_adj_y, pretheta, problem, use_CG, omega_full)
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
def QoIdist(qoi, quad_points, pi_theta_quad, d_area, boxlims, lmbda, V, neg_adj_y, pretheta, problem, use_CG=False, omega_full=None):
    output = np.zeros(len(qoi))
    gauss_evals = np.zeros((len(qoi),quad_points.shape[0]))
    # for each quadrature point:
    for idx in range(quad_points.shape[0]):
        # find Gaussian pi(qoi|theta,y) where theta = the quadrature point
        theta = quad_points[idx,:]
        gauss_evals[:,idx] = QoIdist_fixed_theta(qoi, theta, boxlims, lmbda, V, neg_adj_y, pretheta, problem, use_CG, omega_full)
        # multiply by pi(theta|y) at qpt and area/volume element and add
        output += d_area*pi_theta_quad[idx]*gauss_evals[:,idx]
    return output

## QoI distribution for when QoI is pointwise evaluation
## Finds QoI distribution at a list of locations simultaneously (unlike general code above)
# find pi(x^i|y) for each location i in locs at values u_0_eval of u_0
def posterior_marginals(locs, u_0_eval, quad_points, pi_theta_quad, d_area, lmbda, V, 
                        neg_adj_y, pretheta, problem, omega_full=None):
    output = np.zeros((len(locs),len(u_0_eval)))
    gauss_evals = np.zeros((len(locs),len(u_0_eval),quad_points.shape[0]))
    for idx in range(quad_points.shape[0]):
        theta = quad_points[idx,:]
        posterior,mg,lmbda_new,V_new = ComputePosterior(theta, lmbda, V, neg_adj_y, pretheta, problem, omega_full)
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

def compute_all_qoi_distributions(theta_MAP, thetas, qoi_box, quad_points, pi_theta_quad, 
                                  d_area, qoi_range_bounds, num_qoi_values, low_rank, dim, Vh, Vh2, 
                                  true_initial_condition, neg_adj_y, problem,
                                  use_CG=False, theta_idx_labels=None, *, plot=True, save=False):
    lmbda = low_rank.eigenvalues; V = low_rank.eigenvectors
    pretheta = low_rank.pretheta; omega_full = low_rank.sketching_matrix

    # print QoI(constant 1 function) to test error introduced by finite element approx
    testu0 = dl.interpolate(dl.Constant(1), Vh2).vector()
    print(f"QoI(constant 1 function) = {QoI(testu0, Vh2, qoi_box)}")

    # evaluate pi(qoi|y) at range of qoi values
    qoi_range = np.linspace(qoi_range_bounds[0],qoi_range_bounds[1],num_qoi_values)
    pi_qoi = QoIdist(qoi_range, quad_points, pi_theta_quad, d_area, qoi_box, lmbda, V, neg_adj_y, pretheta, problem, use_CG, omega_full)

    pi_qoi_th_MAP = QoIdist_fixed_theta(qoi_range, theta_MAP, qoi_box, lmbda, V, neg_adj_y, pretheta, problem, use_CG, omega_full)
    pi_qoi_thetas = [0 for idx in thetas]
    for idx in range(len(thetas)):
        pi_qoi_thetas[idx] = QoIdist_fixed_theta(qoi_range, thetas[idx], qoi_box, lmbda, V, neg_adj_y, pretheta, problem, use_CG, omega_full)

    # true QoI
    true_QoI = QoI(true_initial_condition, Vh, qoi_box)

    if theta_idx_labels is None:
        theta_idx_labels = [idx in range(len(thetas))]

    if plot:
        if dim == 2:
            fig, ax = plt.subplots()
            ic = dl.Function(Vh)
            ic.vector()[:] = true_initial_condition
            plt.sca(ax)
            plot_obj = dl.plot(ic)
            ax.set_title("IC and Box Location")    
            rect = Rectangle((qoi_box[0], qoi_box[2]), qoi_box[1]-qoi_box[0], qoi_box[3]-qoi_box[2], edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            fig.colorbar(plot_obj, ax=ax)
        else:
            print("QoI box and initial condition can only be visualized in 2D case")
    # plot distribution of QoI
        plt.figure(figsize=(10,7.2))
        plt.rcParams.update({'font.size': 20})
        plt.plot(qoi_range,pi_qoi_th_MAP,linewidth=3,color='green', label=rf"$\theta^\ast$")
        for idx in range(len(thetas)):
            plt.plot(qoi_range,pi_qoi_thetas[idx],linewidth=3, label=rf"$\theta_{theta_idx_labels[idx]+1}$")
        plt.plot(qoi_range,pi_qoi,linewidth=3,color='black', label=r"marginalized")
        plt.axvline(x=true_QoI, color='black', linestyle="-.", label=r"true qoi")
        plt.title(rf'Posterior Distribution of QoI $q$')
        plt.ylabel(r"$\pi(q|y)$")
        plt.xlabel(r"$q$")
        plt.tight_layout()
        plt.legend()

    if save:
        filename = "images/piQoI.txt"
        header = "q \t\t theta_opt"
        data = np.column_stack((qoi_range, pi_qoi_th_MAP))
        for idx in range(len(thetas)):
            header = header + f" \t\t theta_{theta_idx_labels[idx]+1}"
            data = np.column_stack((data, pi_qoi_thetas[idx]))
        header = header + " \t\t marginalized"
        data = np.column_stack((data, pi_qoi))
        np.savetxt(filename, data, delimiter="\t", header=header, fmt='%10.8f', comments="")

    return pi_qoi, pi_qoi_th_MAP, pi_qoi_thetas, true_QoI