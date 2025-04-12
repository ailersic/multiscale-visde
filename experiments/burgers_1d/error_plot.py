from matplotlib import pyplot as plt
import os
import pathlib

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Garamond",
    "font.size": 12
})

#errors = [0.44, 0.156, 0.068, 0.085, 0.159, 0.063]
errors = [0.323, 0.158, 0.067, 0.068, 0.123, 0.064]
fig, ax = plt.subplots(figsize=(5, 2))
ax.plot([0, 1, 2, 3, 4, 5], errors, marker='o', color='black')
ax.set_xlabel("Microscale state dimension")
ax.set_ylabel("Error")
ax.set_ylim([0.0, 0.4])
ax.grid()
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(CURR_DIR, "error_plot.pdf"), format='pdf')
plt.close()
