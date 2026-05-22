# %%
import dolfin as dl
import os

# %%
def v_boundary(x,on_boundary):
    return on_boundary

def wall_left_right(x, on_boundary):
    return on_boundary and (x[0] < 1e-14 or x[0] > 1.0 - 1e-14)

def wall_front_back(x, on_boundary):
    return on_boundary and (x[1] < 1e-14 or x[1] > 1.0 - 1e-14)

def wall_top_bottom(x, on_boundary):
    return on_boundary and (x[2] < 1e-14 or x[2] > 1.0 - 1e-14)

# Identify building surfaces (everything on boundary that isn't the outer shell)
def buildings(x, on_boundary):
    is_outer_shell = (x[0] < 1e-14 or x[0] > 1.0 - 1e-14 or 
                      x[1] < 1e-14 or x[1] > 1.0 - 1e-14 or 
                      x[2] < 1e-14 or x[2] > 1.0 - 1e-14)
    return on_boundary and not is_outer_shell

def q_boundary(x, on_boundary):
    # Pins pressure at the origin point (0,0,0)
    return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS and x[2] < dl.DOLFIN_EPS
        
def computeVelocityField(mesh):
    Xh = dl.VectorFunctionSpace(mesh,'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = dl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(1e2)
    
    taper = "std::min(1.0, std::min(10*x[1], std::min(10*(1-x[1]), std::min(10*x[2], 10*(1-x[2])))))"
    g_str = ('0.0', 
            '((x[0] < 1e-14) - (x[0] > 1.0 - 1e-14)) * ' + taper, 
            '((x[0] < 1e-14) - (x[0] > 1.0 - 1e-14)) * ' + taper)
    
    g = dl.Expression(g_str, degree=1)

    bcs = []

    # A. DRIVING WALLS: Fixed velocity on Left/Right faces
    bcs.append(dl.DirichletBC(XW.sub(0), g, wall_left_right))

    # B. SLIP WALLS: Zero normal velocity
    # No flow through Front/Back (Y-direction)
    bcs.append(dl.DirichletBC(XW.sub(0).sub(1), dl.Constant(0.0), wall_front_back))
    # No flow through Top/Bottom (Z-direction)
    bcs.append(dl.DirichletBC(XW.sub(0).sub(2), dl.Constant(0.0), wall_top_bottom))

    # C. BUILDINGS: No-slip (all components = 0)
    bcs.append(dl.DirichletBC(XW.sub(0), dl.Constant((0, 0, 0)), buildings))

    # D. PRESSURE PIN: Remove the nullspace
    # Use a small range in the middle to find a node for pressure reference
    bc_p = dl.DirichletBC(XW.sub(1), dl.Constant(0), 
                         "x[0] > 0.4 && x[0] < 0.6 && x[1] > 0.4 && x[1] < 0.6 && x[2] > 0.4 && x[2] < 0.6", 
                         "pointwise")
    bcs.append(bc_p)
    
    vq = dl.Function(XW)
    (v,q) = dl.split(vq)
    (v_test, q_test) = dl.TestFunctions (XW)
    
    def strain(v):
        return dl.sym(dl.grad(v))
    
    F = ( (2./Re)*dl.inner(strain(v),strain(v_test))
         + dl.inner (dl.nabla_grad(v)*v, v_test)
           - (q * dl.div(v_test)) + ( dl.div(v) * q_test) ) * dl.dx
           
    dl.solve(F == 0, vq, bcs, solver_parameters={"newton_solver":
                                         {
                                          "linear_solver": "mumps",
                                        #   "linear_solver": "gmres",
                                        #   "preconditioner": "ilu",
                                          "relative_tolerance":1e-4, 
                                          "maximum_iterations":100,
                                          "report": True}})
        
    return vq #v


# %% Load the mesh
dofs = 3574
mesh = dl.Mesh("meshes/cube_3d_dofs_{0}.xml".format(dofs))
V = dl.FunctionSpace(mesh, "Lagrange", 1)

# %%
# Call the function
velocity_mixed = computeVelocityField(mesh)

velocity = velocity_mixed.split(deepcopy=True)[0]
# Rename for clarity
velocity.rename("velocity", "velocity")

#%%
# Save for Paraview
file = dl.File("velocity_fields/velocity_field_{0}.pvd".format(dofs))
file << velocity

# Save in HDF5
filename = "velocity_fields/velocity_field_{0}.h5".format(dofs)
# If the file exists, remove it first to ensure a fresh start
if os.path.exists(filename):
    try:
        os.remove(filename)
    except OSError:
        print("Warning: File is locked. Try a different filename.")
# The 'with' block automatically handles f.close() for you
with dl.HDF5File(mesh.mpi_comm(), filename, "w") as f:
    f.write(mesh, "/mesh")
    f.write(velocity, "/velocity")

# Save in XML as a backup
dl.File("velocity_field_{0}.xml".format(dofs)) << velocity

#%%
# Save exterior surface of mesh for Paraview
boundary_mesh = dl.BoundaryMesh(mesh, "exterior")

boundary_file = dl.File("velocity_fields/domain_boundary_{0}.pvd".format(dofs))
boundary_file << boundary_mesh

# %%
