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
# logging.getLogger('FFC').setLevel(logging.WARNING)
#logging.getLogger('UFL').setLevel(logging.WARNING)
# dl.set_log_active(False)

import time
import line_profiler
#%load_ext line_profiler
np.random.seed(42)

# %%
# def v_boundary(x,on_boundary):
#     return on_boundary

# def q_boundary(x,on_boundary):
#     return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS
        
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

import dolfin as dl

def v_boundary(x, on_boundary):
    return on_boundary

def q_boundary(x, on_boundary):
    # Pins pressure at the origin point (0,0,0)
    return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS and x[2] < dl.DOLFIN_EPS

def computeVelocityField3D(mesh):
    # 1. Define Function Spaces for 3D
    # VectorFunctionSpace defaults to the dimension of the mesh (3 for a cube)
    Xh = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = dl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(0.1)
    
    # 2. Define the 3D Velocity Expression
    # Component 0 (vx): 0
    # Component 1 (vy): 1 on left, -1 on right
    # Component 2 (vz): 1 on left, -1 on right
    g_str = ('0.0', 
             '(x[0] < 1e-14) - (x[0] > 1.0 - 1e-14)', 
            #  '(x[0] < 1e-14) - (x[0] > 1.0 - 1e-14)')
            '0.0')
    g = dl.Expression(g_str, degree=1)
    
    # 3. Boundary Conditions
    bc1 = dl.DirichletBC(XW.sub(0), g, v_boundary)
    bc2 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')
    bcs = [bc1, bc2]
    
    # 4. Variational Form (The math remains the same, FEniCS handles the dim)
    vq = dl.Function(XW)
    (v, q) = dl.split(vq)
    (v_test, q_test) = dl.TestFunctions(XW)
    
    def strain(v):
        return dl.sym(dl.grad(v))
    
    # Navier-Stokes Weak Form
    F = ( (2./Re)*dl.inner(strain(v), strain(v_test)) 
         + dl.inner(dl.nabla_grad(v)*v, v_test)
         - (q * dl.div(v_test)) 
         + (dl.div(v) * q_test) ) * dl.dx
           
    # 5. Solve
    dl.solve(F == 0, vq, bcs, solver_parameters={"newton_solver":
                                                 {"relative_tolerance": 1e-4, 
                                                  "maximum_iterations": 15,
                                                  "report": True}})
    
    return vq


# %%
# 1. Load the mesh
dofs = 1281
mesh = dl.Mesh("meshes/cube_3d_dofs_{0}.xml".format(dofs))

# mesh = dl.UnitCubeMesh(10, 10, 10)

# 2. Define the Function Space (P1 Lagrange)
V = dl.FunctionSpace(mesh, "Lagrange", 1)

# %%
# Call the function (now returning the full vq)
velocity_mixed = computeVelocityField3D(mesh)

# Extract the velocity component (sub(0) is velocity, sub(1) is pressure)
# Using deepcopy=True creates a clean function object for ParaView
v_plot = velocity_mixed.split(deepcopy=True)[0]

# Rename for clarity in ParaView
v_plot.rename("velocity", "velocity")

# Save
file = dl.File("velocity_results_city_low_Re.pvd")
file << v_plot

# #%%
# # Extract the exterior surface of your mesh
# boundary_mesh = dl.BoundaryMesh(mesh, "exterior")

# # Save it as a separate file
# boundary_file = dl.File("domain_boundary.pvd")
# boundary_file << boundary_mesh

# %%
# 3. Define the Gaussian (centered at (0.2,0.8,0.5))
# Using hippylib's Expression syntax or standard dolfin
center = ((0.2,0.8,0.5))
width = 25.0
gaussian_expr = dl.Expression(
    "exp(-a * (pow(x[0]-x0, 2) + pow(x[1]-y0, 2) + pow(x[2]-z0, 2)))",
    a=width, x0=center[0], y0=center[1], z0=center[2],
    degree=2
)

# 4. Create the function
u = dl.interpolate(gaussian_expr, V)

# %%
# Standard FEniCS plot (requires a working X11 server/WSLg)
dl.plot(u, title="3D Gaussian")
plt.show()
# %%
