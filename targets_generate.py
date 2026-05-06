import numpy as np

# ************ 2D observations **************

# number of points on each side of the two boxes
pts = 4
targets2d = np.zeros((8*pts,2))
targets2d[0:pts,0] = np.linspace(.25, .5, pts+2)[1:-1]
targets2d[0:pts,1] = 0.149
targets2d[pts:2*pts,0] = np.linspace(.25, .5, pts+2)[1:-1]
targets2d[pts:2*pts,1] = 0.401
targets2d[2*pts:3*pts,0] = 0.249
targets2d[2*pts:3*pts,1] = np.linspace(.15, .4, pts+2)[1:-1]
targets2d[3*pts:4*pts,0] = 0.501
targets2d[3*pts:4*pts,1] = np.linspace(.15, .4, pts+2)[1:-1]
targets2d[4*pts:5*pts,0] = np.linspace(.6, .75, pts+2)[1:-1]
targets2d[4*pts:5*pts,1] = 0.599
targets2d[5*pts:6*pts,0] = np.linspace(.6, .75, pts+2)[1:-1]
targets2d[5*pts:6*pts,1] = 0.851
targets2d[6*pts:7*pts,0] = 0.599
targets2d[6*pts:7*pts,1] = np.linspace(.6, .85, pts+2)[1:-1]
targets2d[7*pts:8*pts,0] = 0.751
targets2d[7*pts:8*pts,1] = np.linspace(.6, .85, pts+2)[1:-1]

# Save to .txt
targets_2d_array = np.array(targets2d)
np.savetxt('targets_2d.txt', targets_2d_array, fmt='%.6f', header='x y')


# ************ 3D observations **************

# Building data: (x, y, z, dx, dy, dz)
buildings = [
    (0.2, 0.2, 0.0, 0.2, 0.2, 0.6), # Building 2
    (0.6, 0.2, 0.0, 0.2, 0.2, 0.4), # Building 3
    (0.4, 0.6, 0.0, 0.2, 0.3, 0.8)  # Building 4
]

all_targets = []

for (x, y, z_base, dx, dy, dz) in buildings:
    z_top = z_base + dz
    z_mid = z_base + (dz / 2.0)
    
    # 1. Four Upper Corners (at z_top)
    upper_corners = [
        [x,      y,      z_top],
        [x + dx, y,      z_top],
        [x,      y + dy, z_top],
        [x + dx, y + dy, z_top]
    ]
    
    # 2. Midpoints of the four vertical side edges (at z_mid)
    side_mids = [
        [x,      y,      z_mid],
        [x + dx, y,      z_mid],
        [x,      y + dy, z_mid],
        [x + dx, y + dy, z_mid]
    ]
    
    all_targets.extend(upper_corners)
    all_targets.extend(side_mids)

# Convert to numpy array and save
targets_3d_array = np.array(all_targets)
np.savetxt('targets_3d.txt', targets_3d_array, fmt='%.6f', header='x y z')