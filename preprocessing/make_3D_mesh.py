# %%
import dolfin as dl
import subprocess
import sys
from pathlib import Path
import gmsh

try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    REPO_ROOT = Path.cwd()

MESH_DIR = REPO_ROOT / "meshes"
MESH_DIR.mkdir(parents=True, exist_ok=True)

MSH_PATH = MESH_DIR / "cube_3d.msh"
XML_PATH = MESH_DIR / "cube_3d.xml"

# %% Initialize Gmsh
gmsh.initialize()
gmsh.model.add("3d_city")

# Set mesh size (characteristic length)
# 0.33e-1 -> 25,101 vertices
# 0.5e-1 -> 7,516 vertices
lc = 0.5e-1 

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
gmsh.write(str(MSH_PATH))
subprocess.run(
    ["dolfin-convert", str(MSH_PATH), str(XML_PATH)],
    check=True,
)

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

# %% Load into FEniCS (Dolfin)
mesh = dl.Mesh(str(XML_PATH))
Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
dims = Vh.dim()

print(f"Mesh Cells: {mesh.num_cells()}")
print(f"Degrees of Freedom: {dims}")

final_path = MESH_DIR / f"cube_3d_dofs_{dims}.xml"
XML_PATH.rename(final_path)

# # Optional: Save for Paraview visualization
# file = dl.File("mesh_output.pvd")
# file << mesh
# %%
