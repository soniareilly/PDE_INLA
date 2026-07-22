# These functions are based on the hIPPYlib library, version 3.1.0.
# https://hippylib.github.io


# Changes made:
# diffusivity kappa no longer hardcoded
# adds support for different state and parameter function spaces
# adds support to BilaplacianPrior for gamma = 0
# replaces exact solve with least squares in randomized eigensolver to avoid singularity errors when the rank is low
# replaces Krylov solve with direct Cholesky solve for prior for speed

import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt
import numbers
# import argparse

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *

# no change from hippylib adv_diff application
class SpaceTimePointwiseStateObservation(Misfit):
    def __init__(self, Vh,
                 observation_times,
                 targets,
                 d = None,
                 noise_variance=None):
        
        self.Vh = Vh
        self.observation_times = observation_times
        
        self.B = assemblePointwiseObservation(self.Vh, targets)
        self.ntargets = targets
        
        if d is None:
            self.d = TimeDependentVector(observation_times)
            self.d.initialize(self.B, 0)
        else:
            self.d = d
            
        self.noise_variance = noise_variance
        
        ## TEMP Vars
        self.u_snapshot = dl.Vector()
        self.Bu_snapshot = dl.Vector()
        self.d_snapshot  = dl.Vector()
        self.B.init_vector(self.u_snapshot, 1)
        self.B.init_vector(self.Bu_snapshot, 0)
        self.B.init_vector(self.d_snapshot, 0)
        
    def observe(self, x, obs):        
        obs.zero()
        
        for t in self.observation_times:
            x[STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            obs.store(self.Bu_snapshot, t)
            
    def cost(self, x):
        c = 0
        for t in self.observation_times:
            x[STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            self.d.retrieve(self.d_snapshot, t)
            self.Bu_snapshot.axpy(-1., self.d_snapshot)
            c += self.Bu_snapshot.inner(self.Bu_snapshot)
            
        return c/(2.*self.noise_variance)
    
    def grad(self, i, x, out):
        out.zero()
        if i == STATE:
            for t in self.observation_times:
                x[STATE].retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.d.retrieve(self.d_snapshot, t)
                self.Bu_snapshot.axpy(-1., self.d_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                out.store(self.u_snapshot, t)           
        else:
            pass
            
    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        pass
    
    def apply_ij(self, i,j, direction, out):
        out.zero()
        if i == STATE and j == STATE:
            for t in self.observation_times:
                direction.retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.Bu_snapshot *= 1./self.noise_variance
                self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                out.store(self.u_snapshot, t)
        else:
            pass    



class TimeDependentAD:    
    def __init__(self, mesh, Vh, prior, misfit, simulation_times, diffusivity, wind_velocity, gls_stab):
        self.mesh = mesh
        self.Vh = Vh
        self.prior = prior
        self.misfit = misfit
        # Assume constant timestepping
        self.simulation_times = simulation_times
        dt = simulation_times[1] - simulation_times[0]


        u = dl.TrialFunction(Vh[STATE])
        v = dl.TestFunction(Vh[STATE])
        m_trial = dl.TrialFunction(Vh[PARAMETER])
        
        #kappa = dl.Constant(.001)
        kappa = dl.Constant(diffusivity)
        dt_expr = dl.Constant(dt)
        
        r_trial = u + dt_expr*( -dl.div(kappa*dl.grad(u))+ dl.inner(wind_velocity, dl.grad(u)) )
        r_test  = v + dt_expr*( -dl.div(kappa*dl.grad(v))+ dl.inner(wind_velocity, dl.grad(v)) )

        
        h = dl.CellDiameter(mesh)
        vnorm = dl.sqrt(dl.inner(wind_velocity, wind_velocity))
        if gls_stab:
            tau = ufl.min_value((h*h)/(dl.Constant(2.)*kappa), h/vnorm )
        else:
            tau = dl.Constant(0.)
                            
        self.M = dl.assemble( dl.inner(u,v)*dl.dx )
        self.M_stab = dl.assemble( dl.inner(u, v+tau*r_test)*dl.dx )
        self.M_stab_mixed = dl.assemble( dl.inner(m_trial, v+tau*r_test)*dl.dx )
        self.Mt_stab = dl.assemble( dl.inner(u+tau*r_trial,v)*dl.dx )
        Nvarf  = (dl.inner(kappa * dl.grad(u), dl.grad(v)) + dl.inner(wind_velocity, dl.grad(u))*v )*dl.dx
        Ntvarf  = (dl.inner(kappa *dl.grad(v), dl.grad(u)) + dl.inner(wind_velocity, dl.grad(v))*u )*dl.dx
        self.N  = dl.assemble( Nvarf )
        self.Nt = dl.assemble(Ntvarf)
        stab = dl.assemble( tau*dl.inner(r_trial, r_test)*dl.dx)
        self.L = self.M + dt*self.N + stab
        self.Lt = self.M + dt*self.Nt + stab
        
        self.solver  = PETScLUSolver( self.mesh.mpi_comm() )
        self.solver.set_operator( dl.as_backend_type(self.L) )
        self.solvert = PETScLUSolver( self.mesh.mpi_comm() ) 
        self.solvert.set_operator(dl.as_backend_type(self.Lt) )
                        
        # Part of model public API
        self.gauss_newton_approx = False
                    
    def generate_vector(self, component = "ALL"):
        if component == "ALL":
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.M, 0)
            m = dl.Vector()
            self.prior.init_vector(m,0)
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.M, 0)
            return [u, m, p]
        elif component == STATE:
            u = TimeDependentVector(self.simulation_times)
            u.initialize(self.M, 0)
            return u
        elif component == PARAMETER:
            m = dl.Vector()
            self.prior.init_vector(m,0)
            return m
        elif component == ADJOINT:
            p = TimeDependentVector(self.simulation_times)
            p.initialize(self.M, 0)
            return p
        else:
            raise
    
    def init_parameter(self, m):
        self.prior.init_vector(m,0)
        
          
    def cost(self, x):
        Rdx = dl.Vector()
        self.prior.init_vector(Rdx,0)
        dx = x[PARAMETER] - self.prior.mean
        self.prior.R.mult(dx, Rdx)
        reg = .5*Rdx.inner(dx)
        
        misfit = self.misfit.cost(x)
                
        return [reg+misfit, reg, misfit]
    
    def solveFwd(self, out, x):
        out.zero()
        u = dl.Vector()
        rhs = dl.Vector()
        self.M.init_vector(rhs, 0)
        self.M.init_vector(u, 0)
        is_first_step = True
        for t in self.simulation_times[1:]:
            if is_first_step:
                self.M_stab_mixed.mult(x[PARAMETER], rhs)
                is_first_step = False
            else:
                self.M_stab.mult(uold, rhs)
            self.solver.solve(u, rhs)
            out.store(u, t)
            uold = u.copy()
    
    def solveAdj(self, out, x):
        
        grad_state = TimeDependentVector(self.simulation_times)
        grad_state.initialize(self.M, 0)
        self.misfit.grad(STATE, x, grad_state)
        
        out.zero()
        
        pold = dl.Vector()
        self.M.init_vector(pold,0)
            
        p = dl.Vector()
        self.M.init_vector(p,0)
        
        rhs = dl.Vector()
        self.M.init_vector(rhs,0)
        
        grad_state_snap = dl.Vector()
        self.M.init_vector(grad_state_snap,0)

          
        for t in self.simulation_times[::-1]:
            self.Mt_stab.mult(pold,rhs)
            grad_state.retrieve(grad_state_snap, t)
            rhs.axpy(-1., grad_state_snap)
            self.solvert.solve(p, rhs)
            pold = p
            out.store(p, t)            
            
    def evalGradientParameter(self,x, mg, misfit_only=False):
        self.prior.init_vector(mg,1)
        if misfit_only == False:
            dm = x[PARAMETER] - self.prior.mean
            self.prior.R.mult(dm, mg)
        else:
            mg.zero()
        
        p0 = dl.Vector()
        self.M.init_vector(p0,0)
        x[ADJOINT].retrieve(p0, self.simulation_times[1])

        grad_param = dl.Vector()
        self.prior.init_vector(grad_param, 0)
        self.M_stab_mixed.transpmult(p0, grad_param)
        mg.axpy(-1., grad_param)
        
        g = dl.Vector()
        self.prior.init_vector(g,1)
        
        self.prior.Msolver.solve(g,mg)
        
        grad_norm = g.inner(mg)
        
        return grad_norm
        
    
    def setPointForHessianEvaluations(self, x, gauss_newton_approx=False):
        """
        Specify the point x = [u,a,p] at which the Hessian operator (or the Gauss-Newton approximation)
        need to be evaluated.
        
        Nothing to do since the problem is linear
        """
        self.gauss_newton_approx = gauss_newton_approx
        return

        
    def solveFwdIncremental(self, sol, rhs):
        sol.zero()
        uold = dl.Vector()
        u = dl.Vector()
        Muold = dl.Vector()
        myrhs = dl.Vector()
        self.M.init_vector(uold, 0)
        self.M.init_vector(u, 0)
        self.M.init_vector(Muold, 0)
        self.M.init_vector(myrhs, 0)

        for t in self.simulation_times[1::]:
            self.M_stab.mult(uold, Muold)
            rhs.retrieve(myrhs, t)
            myrhs.axpy(1., Muold)
            self.solver.solve(u, myrhs)
            sol.store(u,t)
            uold = u


        
    def solveAdjIncremental(self, sol, rhs):
        sol.zero()
        pold = dl.Vector()
        p = dl.Vector()
        Mpold = dl.Vector()
        myrhs = dl.Vector()
        self.M.init_vector(pold, 0)
        self.M.init_vector(p, 0)
        self.M.init_vector(Mpold, 0)
        self.M.init_vector(myrhs, 0)

        for t in self.simulation_times[::-1]:
            self.Mt_stab.mult(pold,Mpold)
            rhs.retrieve(myrhs, t)
            Mpold.axpy(1., myrhs)
            self.solvert.solve(p, Mpold)
            pold = p
            sol.store(p, t)  
            
    
    def applyC(self, dm, out):
        out.zero()
        myout = dl.Vector()
        self.M.init_vector(myout, 0)
        self.M_stab_mixed.mult(dm,myout)
        myout *= -1.
        t = self.simulation_times[1]
        out.store(myout,t)
        
        myout.zero()
        for t in self.simulation_times[2:]:
            out.store(myout,t)
    
    def applyCt(self, dp, out):
        t = self.simulation_times[1]
        dp0 = dl.Vector()
        self.M.init_vector(dp0,0)
        dp.retrieve(dp0, t)
        dp0 *= -1.
        self.M_stab_mixed.transpmult(dp0, out)

    
    def applyWuu(self, du, out):
        out.zero()
        self.misfit.apply_ij(STATE, STATE, du, out)

    
    def applyWum(self, dm, out):
        out.zero()

    
    def applyWmu(self, du, out):
        out.zero()
    
    def applyR(self, dm, out):
        self.prior.R.mult(dm,out)
    
    def applyWmm(self, dm, out):
        out.zero()
        
    def exportState(self, x, filename, varname):
        out_file = dl.XDMFFile(self.Vh[STATE].mesh().mpi_comm(), filename)
        out_file.parameters["functions_share_mesh"] = True
        out_file.parameters["rewrite_function_mesh"] = False
        ufunc = dl.Function(self.Vh[STATE], name=varname)
        t = self.simulation_times[0]
        out_file.write(vector2Function(x[PARAMETER], self.Vh[STATE], name=varname),t)
        for t in self.simulation_times[1:]:
            x[STATE].retrieve(ufunc.vector(), t)
            out_file.write(ufunc, t)
    
# no change from hippylib adv_diff application
def v_boundary(x,on_boundary):
    return on_boundary

# no change from hippylib adv_diff application
def q_boundary(x,on_boundary):
    return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS
        
# no change from hippylib adv_diff application
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
        
    return v

# replaced np.linalg.solve to np.linalg.lstsq to avoid singularity errors when the rank is too low
def singlePassG(A, B, Binv, Omega,k, s = 1, check = False):
    nvec  = Omega.nvec()
    
    assert(nvec >= k )
    
    Ybar = MultiVector(Omega[0], nvec)
    Y_pr = MultiVector(Omega)
    Q = MultiVector(Omega)
    for i in range(s):
        Y_pr.swap(Q)
        MatMvMult(A, Y_pr, Ybar)
        MatMvMult(Solver2Operator(Binv), Ybar, Q)
    
    BQ, _ = Q.Borthogonalize(B)
    
    Xt = Y_pr.dot_mv(BQ)
    Wt = Ybar.dot_mv(Q)
    Tt = np.linalg.lstsq(Xt, Wt, rcond=None)[0]
                
    T = .5*Tt + .5*Tt.T
        
    d, V = np.linalg.eigh(T)
    sort_perm = d.argsort()
        
    sort_perm = sort_perm[::-1]
    d = d[sort_perm[0:k]]
    V = V[:, sort_perm[0:k]] 
        
    U = MultiVector(Omega[0], k)
    MvDSmatMult(Q, V, U)
    
    if check:
        check_g(A,B, U, d)
        
    return d, U

# no change from prior.py
class _Prior:
    """
    Abstract class to describe the prior model.
    Concrete instances of a :code:`_Prior class` should expose
    the following attributes and methods.
    
    Attributes:

    - :code:`R`:       an operator to apply the regularization/precision operator.
    - :code:`Rsolver`: an operator to apply the inverse of the regularization/precision operator.
    - :code:`M`:       the mass matrix in the control space.
    - :code:`mean`:    the prior mean.
    
    Methods:

    - :code:`init_vector(self,x,dim)`: Inizialize a vector :code:`x` to be compatible with the range/domain of :code:`R`
      If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
      white noise used for sampling.
      
    - :code:`sample(self, noise, s, add_mean=True)`: Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample s from the prior.
      If :code:`add_mean==True` add the prior mean value to :code:`s`.
    """ 
               
    def trace(self, method="Exact", tol=1e-1, min_iter=20, max_iter=100, r = 200):
        """
        Compute/estimate the trace of the prior covariance operator.
        
        - If :code:`method=="Exact"` we compute the trace exactly by summing the diagonal entries of :math:`R^{-1}M`.
          This requires to solve :math:`n` linear system in :math:`R` (not scalable, but ok for illustration purposes).
          
        - If :code:`method=="Estimator"` use the trace estimator algorithms implemeted in the class :code:`TraceEstimator`.
          :code:`tol` is a relative bound on the estimator standard deviation. In particular, we used enough samples in the
          Estimator such that the standard deviation of the estimator is less then :code:`tol`:math:`tr(\\mbox{Prior})`.
          :code:`min_iter` and :code:`max_iter` are the lower and upper bound on the number of samples to be used for the
          estimation of the trace. 
        """
        op = _RinvM(self.Rsolver, self.M)
        if method == "Exact":
            marginal_variance = dl.Vector(self.R.mpi_comm())
            self.init_vector(marginal_variance,0)
            get_diagonal(op, marginal_variance)
            return marginal_variance.sum()
        elif method == "Estimator":
            tr_estimator = TraceEstimator(op, False, tol)
            tr_exp, tr_var = tr_estimator(min_iter, max_iter)
            return tr_exp
        elif method == "Randomized":
            dummy = dl.Vector(self.R.mpi_comm())
            self.init_vector(dummy,0)
            Omega = MultiVector(dummy, r)
            parRandom.normal(1., Omega)
            d, _ = doublePassG(Solver2Operator(self.Rsolver),
                               Solver2Operator(self.Msolver),
                               Operator2Solver(self.M),
                               Omega, r, s = 1, check = False )
            return d.sum()
        else:
            raise NameError("Unknown method")
        
    def pointwise_variance(self, method, k = 1000000, r = 200):
        """
        Compute/estimate the prior pointwise variance.
        
        - If :code:`method=="Exact"` we compute the diagonal entries of :math:`R^{-1}` entry by entry. 
          This requires to solve :math:`n` linear system in :math:`R` (not scalable, but ok for illustration purposes).
        """
        pw_var = dl.Vector(self.R.mpi_comm())
        self.init_vector(pw_var,0)
        if method == "Exact":
            get_diagonal(Solver2Operator(self.Rsolver, init_vector=self.init_vector), pw_var)
        elif method == "Estimator":
            estimate_diagonal_inv2(self.Rsolver, k, pw_var)
        elif method == "Randomized":
            Omega = MultiVector(pw_var, r)
            parRandom.normal(1., Omega)
            d, U = doublePass(Solver2Operator(self.Rsolver),
                               Omega, r, s = 1, check = False )
            
            for i in np.arange(U.nvec()):
                pw_var.axpy(d[i], U[i]*U[i])
        else:
            raise NameError("Unknown method")
        
        return pw_var
        
    def cost(self,m):
        d = self.mean.copy()
        d.axpy(-1., m)
        Rd = dl.Vector(self.R.mpi_comm())
        self.init_vector(Rd,0)
        self.R.mult(d,Rd)
        return .5*Rd.inner(d)
    
    def grad(self,m, out):
        d = m.copy()
        d.axpy(-1., self.mean)
        self.R.mult(d,out)

    def init_vector(self,x,dim):
        raise NotImplementedError("Child class should implement method init_vector")

    def sample(self, noise, s, add_mean=True):
        raise NotImplementedError("Child class should implement method sample")

    def getHessianPreconditioner(self):
        " Return the preconditioner for Newton-CG "
        return self.Rsolver

# no change
class _BilaplacianR:
    """
    Operator that represent the action of the regularization/precision matrix
    for the Bilaplacian prior.
    """
    def __init__(self, A, Msolver):
        self.A = A
        self.Msolver = Msolver

        self.help1, self.help2 = dl.Vector(self.A.mpi_comm()), dl.Vector(self.A.mpi_comm())
        self.A.init_vector(self.help1, 0)
        self.A.init_vector(self.help2, 1)
        
    def init_vector(self,x, dim):
        self.A.init_vector(x,1)
        
    def mpi_comm(self):
        return self.A.mpi_comm()
        
    def mult(self,x,y):
        self.A.mult(x, self.help1)
        self.Msolver.solve(self.help2, self.help1)
        self.A.mult(self.help2, y)
        
# no change
class _BilaplacianRsolver():
    """
    Operator that represent the action of the inverse the regularization/precision matrix
    for the Bilaplacian prior.
    """
    def __init__(self, Asolver, M):
        self.Asolver = Asolver
        self.M = M
        
        self.help1, self.help2 = dl.Vector(self.M.mpi_comm()), dl.Vector(self.M.mpi_comm())
        self.init_vector(self.help1, 0)
        self.init_vector(self.help2, 0)
        
    def init_vector(self,x, dim):
        self.M.init_vector(x,1)
        
    def solve(self,x,b):
        nit = self.Asolver.solve(self.help1, b)
        self.M.mult(self.help1, self.help2)
        nit += self.Asolver.solve(x, self.help2)
        return nit

# modified to use Cholesky instead of Krylov in prior solve
class SqrtPrecisionPDE_Prior(_Prior):
    """
    This class implement a prior model with covariance matrix
    :math:`C = A^{-1} M A^-1`,
    where A is the finite element matrix arising from discretization of sqrt_precision_varf_handler
    
    """
    
    def __init__(self, Vh, sqrt_precision_varf_handler, mean=None, rel_tol=1e-12, max_iter=1000):
        """
        Construct the prior model.
        Input:

        - :code:`Vh`:              the finite element space for the parameter
        - :code:sqrt_precision_varf_handler: the PDE representation of the  sqrt of the covariance operator
        - :code:`mean`:            the prior mean
        """

        self.Vh = Vh
        
        trial = dl.TrialFunction(Vh)
        test  = dl.TestFunction(Vh)
        
        varfM = dl.inner(trial,test)*dl.dx       
        self.M = dl.assemble(varfM)
        self.Msolver = PETScLUSolver( self.Vh.mesh().mpi_comm() )
        self.Msolver.set_operator( dl.as_backend_type(self.M) )
        
        self.A = dl.assemble( sqrt_precision_varf_handler(trial, test) )        
        self.Asolver = PETScLUSolver( self.Vh.mesh().mpi_comm() )
        self.Asolver.set_operator( dl.as_backend_type(self.A) )
        
        old_qr = dl.parameters["form_compiler"]["quadrature_degree"]
        dl.parameters["form_compiler"]["quadrature_degree"] = -1
        qdegree = 2*Vh._ufl_element.degree()
        metadata = {"quadrature_degree" : qdegree}


        representation_old = dl.parameters["form_compiler"]["representation"]
        dl.parameters["form_compiler"]["representation"] = "quadrature"
            
        num_sub_spaces = Vh.num_sub_spaces()
        if num_sub_spaces <= 1: #SCALAR PARAMETER
            element = dl.FiniteElement("Quadrature", Vh.mesh().ufl_cell(), qdegree, quad_scheme="default")
        else: #Vector FIELD PARAMETER
            element = dl.VectorElement("Quadrature", Vh.mesh().ufl_cell(),
                                       qdegree, dim=num_sub_spaces, quad_scheme="default")
        Qh = dl.FunctionSpace(Vh.mesh(), element)
            
        ph = dl.TrialFunction(Qh)
        qh = dl.TestFunction(Qh)
        Mqh = dl.assemble(dl.inner(ph,qh)*dl.dx(metadata=metadata))
        if num_sub_spaces <= 1:
            one_constant = dl.Constant(1.)
        else:
            one_constant = dl.Constant( tuple( [1.]*num_sub_spaces) )
        ones = dl.interpolate(one_constant, Qh).vector()
        dMqh = Mqh*ones
        Mqh.zero()
        dMqh.set_local( ones.get_local() / np.sqrt(dMqh.get_local() ) )
        Mqh.set_diagonal(dMqh)
        MixedM = dl.assemble(dl.inner(ph,test)*dl.dx(metadata=metadata))
        self.sqrtM = MatMatMult(MixedM, Mqh)

        dl.parameters["form_compiler"]["quadrature_degree"] = old_qr
        dl.parameters["form_compiler"]["representation"] = representation_old
                             
        self.R = _BilaplacianR(self.A, self.Msolver)      
        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)
         
        self.mean = mean
        
        if self.mean is None:
            self.mean = dl.Vector(self.R.mpi_comm())
            self.init_vector(self.mean, 0)
     
    def init_vector(self,x,dim):
        """
        Inizialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.

        If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """
        if dim == "noise":
            self.sqrtM.init_vector(x, 1)
        else:
            self.A.init_vector(x,dim)   
        
    def sample(self, noise, s, add_mean=True):
        """
        Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s` from the prior.

        If :code:`add_mean == True` add the prior mean value to :code:`s`.
        """
        rhs = self.sqrtM*noise
        self.Asolver.solve(s, rhs)
        
        if add_mean:
            s.axpy(1., self.mean)

# robin_coeff now accepts gamma = 0
def BiLaplacianPrior(Vh, gamma, delta, Theta = None, mean=None, rel_tol=1e-12, max_iter=1000, robin_bc=False):
    if isinstance(gamma, numbers.Number):
        gamma = dl.Constant(gamma)
        
    if isinstance(delta, numbers.Number):
        delta = dl.Constant(delta)

    
    def sqrt_precision_varf_handler(trial, test): 
        if Theta == None:
            varfL = dl.inner(dl.grad(trial), dl.grad(test))*dl.dx
        else:
            varfL = dl.inner( Theta*dl.grad(trial), dl.grad(test))*dl.dx
        
        varfM = dl.inner(trial,test)*dl.dx
        
        varf_robin = dl.inner(trial,test)*dl.ds
        
        if robin_bc:
            robin_coeff = dl.sqrt(delta*gamma)/dl.Constant(1.42) # rewritten so that gamma=0 does not give nan
        else:
            robin_coeff = dl.Constant(0.)
        
        return gamma*varfL + delta*varfM + robin_coeff*varf_robin
    
    return SqrtPrecisionPDE_Prior(Vh, sqrt_precision_varf_handler, mean, rel_tol, max_iter)
