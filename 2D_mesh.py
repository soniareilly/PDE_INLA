# %%
import dolfin as dl
import math
import numpy as np
import sys
import os
import gmsh

# %%
gmsh.initialize()

gmsh.model.add("adv_diff")

lc =9e-2
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
gmsh.model.geo.addPoint(0, 1, 0, lc, 4)

gmsh.model.geo.addPoint(.25, .15, 0, lc, 5)
gmsh.model.geo.addPoint(.5, .15, 0, lc, 6)
gmsh.model.geo.addPoint(.5, .4, 0, lc, 7)
gmsh.model.geo.addPoint(.25, .4, 0, lc, 8)

gmsh.model.geo.addPoint(.6, .6, 0, lc, 9)
gmsh.model.geo.addPoint(.75, .6, 0, lc, 10)
gmsh.model.geo.addPoint(.75, .85, 0, lc, 11)
gmsh.model.geo.addPoint(.6, .85, 0, lc, 12)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)

gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(6, 7, 6)
gmsh.model.geo.addLine(7, 8, 7)
gmsh.model.geo.addLine(8, 5, 8)

gmsh.model.geo.addLine(9, 10, 9)
gmsh.model.geo.addLine(10, 11, 10)
gmsh.model.geo.addLine(11, 12, 11)
gmsh.model.geo.addLine(12, 9, 12)

gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 2)
gmsh.model.geo.addCurveLoop([9, 10, 11, 12], 3)

gmsh.model.geo.addPlaneSurface([1, 2, 3], 1)

gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(1, [1, 2, 4], 5)
gmsh.model.addPhysicalGroup(2, [1], name="AdvDiff")

gmsh.model.mesh.generate(2)

#Vh = dl.FunctionSpace(gmsh.model.mesh, "Lagrange", 1)
#gmsh.write("adv_diff_res_{0}.msh".format(lc))
gmsh.option.setNumber("Mesh.MshFileVersion",2.2)  
gmsh.write("adv_diff.msh")
os.system("dolfin-convert {0} {1}".format("adv_diff.msh", "adv_diff.xml"))

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# gmsh.finalize()

# %%
mesh = dl.refine( dl.Mesh("adv_diff.xml") )
Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
dims = Vh.dim()
print(dims)

# %%
os.system("mv adv_diff.xml adv_diff_dofs_{0}.xml".format(dims))

# %%
# refine each cell by a factor of 2

gmsh.model.mesh.refine()
gmsh.option.setNumber("Mesh.MshFileVersion",2.2)  
gmsh.write("adv_diff.msh")
os.system("dolfin-convert {0} {1}".format("adv_diff.msh", "adv_diff.xml"))

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

# %%
mesh = dl.refine( dl.Mesh("adv_diff.xml") )
Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
dims = Vh.dim()
print(dims)

# %%
os.system("mv adv_diff.xml adv_diff_dofs_{0}.xml".format(dims))
# %%
