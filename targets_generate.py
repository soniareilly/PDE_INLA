import numpy as np
import matplotlib.pyplot as plt

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
    (0.2, 0.2, 0.0, 0.2, 0.2, 0.6), # Building 1
    (0.6, 0.2, 0.0, 0.2, 0.2, 0.4), # Building 2
    (0.4, 0.6, 0.0, 0.2, 0.3, 0.8)  # Building 3
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
    
    # 3. Center of the top half of each side of the buildings
    z_top_half = z_base + (0.75 * dz)
    top_half_sides = [
        [x + (dx / 2.0), y,              z_top_half], # Front face top
        [x + (dx / 2.0), y + dy,         z_top_half], # Back face top
        [x,              y + (dy / 2.0), z_top_half], # Left face top
        [x + dx,         y + (dy / 2.0), z_top_half]  # Right face top
    ]
    
    # 4. Center of the bottom half of each side of the buildings
    z_bottom_half = z_base + (0.25 * dz)
    bottom_half_sides = [
        [x + (dx / 2.0), y,              z_bottom_half], # Front face bottom
        [x + (dx / 2.0), y + dy,         z_bottom_half], # Back face bottom
        [x,              y + (dy / 2.0), z_bottom_half], # Left face bottom
        [x + dx,         y + (dy / 2.0), z_bottom_half]  # Right face bottom
    ]
    
    # 5. Center of the roof of each building
    roof_center = [
        [x + (dx / 2.0), y + (dy / 2.0), z_top]
    ]
    
    # Extend the main list with all coordinates
    all_targets.extend(upper_corners)
    all_targets.extend(side_mids)
    all_targets.extend(top_half_sides)
    all_targets.extend(bottom_half_sides)
    all_targets.extend(roof_center)

# Convert to numpy array and save
targets_3d_array = np.array(all_targets)
np.savetxt('targets_3d.txt', targets_3d_array, fmt='%.6f', header='x y z')

# ==========================================
# NEW: 3D MATPLOTLIB PLOTTING CODE
# ==========================================

# Slice the coordinates directly from the array we just generated
x_coords = targets_3d_array[:, 0]
y_coords = targets_3d_array[:, 1]
z_coords = targets_3d_array[:, 2]

# Initialize 3D plotting arena
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')

# Render the points
ax.scatter(x_coords, y_coords, z_coords, c='crimson', marker='o', s=30, label='Simulation Targets')

# Visual labeling and formatting
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Building Target Points Visualization')
ax.legend()

# Keeps the spatial domain's aspect ratio square so the heights aren't distorted
ax.set_box_aspect([1, 1, 1])
plt.show()