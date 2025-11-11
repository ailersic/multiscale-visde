import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pathlib
import os

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 28
})

# 1. Create a ScalarMappable
cbartype = "state"  # Change this to "error" or "state" as needed
if cbartype == "state":
    cmap = cm.turbo
    norm = plt.Normalize(vmin=-0.22, vmax=1.06)
    label = "state"
else:
    cmap = cm.afmhot
    norm = plt.Normalize(vmin=0, vmax=0.18)
    label = "error"
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# 2. Create a new figure and axes for the colorbar
fig_colorbar = plt.figure(figsize=(12, 2))
ax_colorbar = fig_colorbar.add_axes([0.1, 0.1, 0.8, 0.15])

# 3. Create the colorbar
cbar = plt.colorbar(sm, cax=ax_colorbar, orientation='horizontal')

# 4. Customize the colorbar
#cbar.set_label('Colorbar Label')
#cbar.ax.tick_params(labelsize=10)

# 5. Show or save the colorbar
plt.show()
fig_colorbar.savefig(os.path.join(CURR_DIR, f"{cbartype}_colorbar.pdf"), bbox_inches='tight', format='pdf')