import dolfin as dl
import numpy as np
import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *
from hippylib_changes import *
from hyperparam_marginal import *

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
def QoIdist_fixed_theta(qoi, theta, boxlims, lmbda, V, neg_adj_y, pretheta, problem):
    output = np.zeros(len(qoi))
    # find Gaussian pi(qoi|theta,y)
    prior = BiLaplacianPrior(problem.Vh[PARAMETER], theta[0]*theta[1], theta[1], robin_bc=True)
    prior.mean = dl.interpolate(dl.Constant(0.), problem.Vh[PARAMETER]).vector()
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