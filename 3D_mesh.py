# Adapted from Gemini-generated code

# %%
import dolfin as dl
import os
import sys
import gmsh

# %% Initialize Gmsh
gmsh.initialize()
gmsh.model.add("3d_city")

# Set mesh size (characteristic length)
lc = 0.25e-1 

# Using OpenCASCADE kernel for easy 3D primitives
occ = gmsh.model.occ

# 1. Create the main domain (the large cube)
# addBox(x, y, z, dx, dy, dz, tag)
occ.addBox(0, 0, 0, 1, 1, 1, 1)

# 2. Create the three internal boxes to be used as "cutouts"
occ.addBox(0.2, 0.2, 0.0, 0.2, 0.2, 0.6, 2)
occ.addBox(0.6, 0.2, 0.0, 0.2, 0.2, 0.4, 3)
occ.addBox(0.4, 0.6, 0.0, 0.2, 0.3, 0.8, 4)

# 3. Perform Boolean Subtraction
# cut([(dimension, tag_of_object)], [(dimension, tag_of_tool)])
# Dimension 3 represents a Volume
occ.cut([(3, 1)], [(3, 2), (3, 3), (3, 4)])

# Synchronize to transfer OCC geometry to the Gmsh model
occ.synchronize()

# %% Physical Groups
# Assign a physical group to the resulting volume for FEniCS
# After a cut, the remaining volume tag might change, so we fetch all volumes
volumes = gmsh.model.getEntities(3)
gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], name="Domain")

# Set global mesh size
gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
gmsh.option.setNumber("Mesh.MeshSizeMax", lc)

# %% Generate 3D Mesh
gmsh.model.mesh.generate(3)

# Export to legacy format for dolfin-convert
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("cube_3d.msh")

# Convert .msh to .xml (Standard FEniCS workflow)
os.system("dolfin-convert cube_3d.msh cube_3d.xml")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

# %% Load into FEniCS (Dolfin)
mesh = dl.Mesh("cube_3d.xml")
Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
dims = Vh.dim()

print(f"Mesh Cells: {mesh.num_cells()}")
print(f"Degrees of Freedom: {dims}")

os.system("mv cube_3d.xml cube_3d_dofs_{0}.xml".format(dims))

# # Optional: Save for Paraview visualization
# file = dl.File("mesh_output.pvd")
# file << mesh
# %%
