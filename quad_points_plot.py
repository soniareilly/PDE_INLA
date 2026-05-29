#%% 
import numpy as np
import matplotlib.pyplot as plt

#%%
quad_points = np.loadtxt("images/quad_points_3D.txt", delimiter=",", skiprows=1)
theta_MAP = np.array([3.45302293e-02, 3.39541763e+01, 9.75111448e-03])

#%%
plt.rcParams.update({
    "font.family": "serif",       
    "mathtext.fontset": "cm",
    "font.size": 18,
    "figure.dpi": 300             
})

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(projection='3d')
ax.computed_zorder = False

# white background
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# light gray grid lines
grid_style = {'color': (0.9, 0.9, 0.9, 1.0), 'linewidth': 0.5}
ax.xaxis._axinfo["grid"].update(grid_style)
ax.yaxis._axinfo["grid"].update(grid_style)
ax.zaxis._axinfo["grid"].update(grid_style)

# black quadrature points
ax.scatter(quad_points[:, 0], quad_points[:, 2], quad_points[:, 1], 
           color='black', s=20, edgecolors='none', depthshade=True, zorder=1, label='quad. pts.')

# three green points 
green_shades = ['#b3e600', '#00e600', '#009900'] 
ax.scatter(0.02, 0.01, 60, color=green_shades[0], s=95, edgecolors='none', depthshade=True, label=r'$\theta_1$') 
ax.scatter(0.04, 0.01, 30, color=green_shades[1], s=95, edgecolors='none', depthshade=True, label=r'$\theta_2$') 
ax.scatter(0.08, 0.01, 15, color=green_shades[2], s=95, edgecolors='none', depthshade=True, label=r'$\theta_3$')

# red MAP point 
ax.scatter(theta_MAP[0], theta_MAP[2], theta_MAP[1], 
           color='red', s=190, edgecolors='none', depthshade=False, alpha=1.0, zorder=3, label=r'$\theta_{\mathrm{MAP}}$') 

# axis labels
ax.set_xlabel(r'$\gamma$', fontsize=22, labelpad=14)
ax.set_ylabel(r'$\sigma$', fontsize=22, labelpad=30)
# ax.set_zlabel(r'$\delta$', fontsize=22, labelpad=30)
ax.text2D(1.06, 0.6, r'$\delta$', transform=ax.transAxes, fontsize=22, 
          va='center', ha='center', rotation=0)

# limit the number of ticks
ax.xaxis.get_major_locator().set_params(nbins=5)
ax.yaxis.get_major_locator().set_params(nbins=5)
ax.zaxis.get_major_locator().set_params(nbins=5)

ax.xaxis.set_tick_params(pad=4)
ax.yaxis.set_tick_params(pad=0)
ax.zaxis.set_tick_params(pad=8)

# force the sigma (Y) tick text alignment to the right/bottom-right
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_horizontalalignment('left')
    tick.label1.set_verticalalignment('top')

# ax.legend(loc='upper right', bbox_to_anchor=(0.3, 0.95), frameon=True, 
#           facecolor='white', edgecolor='none', framealpha=0.9)

fig.subplots_adjust(left=0.05, right=0.83, bottom=0.05, top=0.95)

# %%
fig.savefig("images/quad_points_3D.pdf")
# %%
