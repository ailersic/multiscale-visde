import os
import pickle as pkl
import pathlib
import matplotlib.pyplot as plt
import numpy as np

from datasets.cylinder_2d.load_data import load_data
from experiments_refac.cylinder_2d.utils import CaseConfig, DEFAULT_CONFIG

#plt.rcParams.update({'font.size': 16})
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
TRAIN_VAL_TEST = "test"
compplot_out_dir = os.path.join(CURR_DIR, "..", "plot_dns")
pathlib.Path(compplot_out_dir).mkdir(parents=True, exist_ok=True)

config = DEFAULT_CONFIG
data = load_data(config.data_file)

mu = data[f"{TRAIN_VAL_TEST}_mu"].cpu().detach().numpy()
t = data[f"{TRAIN_VAL_TEST}_t"].cpu().detach().numpy()
x = data[f"{TRAIN_VAL_TEST}_x"].cpu().detach().numpy()
f = data[f"{TRAIN_VAL_TEST}_f"].cpu().detach().numpy()

n_traj = mu.shape[0]
n_tsteps = t.shape[1]
x_shape = x.shape[2:]

dim_z_macro = 2 * 32 * 8
sigma = int(np.sqrt(np.prod(x.shape[2:]) // dim_z_macro)+0.5)

# open "cylinder_flow_results.npz" file
results_file = os.path.join(CURR_DIR, 'cylinder_flow_results.npz')
if not os.path.exists(results_file):
    raise FileNotFoundError(f"Results file {results_file} not found.")
results = np.load(results_file)
u_resamp = results['u_resamp']

x_pred = np.expand_dims(np.transpose(u_resamp, (0, 3, 1, 2)), 0)  # [traj, time, component, y, x]
x_true = x
print(f"z_pred shape: {x_pred.shape}")
print(f"z_true shape: {x_true.shape}")

x_pred = x_pred.reshape(n_traj, n_tsteps, -1)
x_true = x_true.reshape(n_traj, n_tsteps, -1)

norm_err = np.zeros((n_traj, n_tsteps))

for i in range(n_traj):
    for j in range(n_tsteps):
        norm_err[i, j] = np.linalg.norm(x_pred[i, j, :] - x_true[i, j, :]) / np.linalg.norm(x_true[i, j, :])

    tsamples = [0, n_tsteps//4, n_tsteps//2, 3*n_tsteps//4, n_tsteps-1]
    cmap = plt.get_cmap("turbo")

    if i == 0:
        for j_plot, j_t in enumerate(tsamples):
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.imshow(x_pred[i, j_t].reshape(x_shape)[0], cmap=cmap, vmin=-1, vmax=1.5)
            plt.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(compplot_out_dir, f"test_{i}_{j_plot}_pred_0.pdf"), format='pdf')
            fig.show()

            fig, ax = plt.subplots(figsize=(5, 2))
            ax.imshow(x_pred[i, j_t].reshape(x_shape)[1], cmap=cmap, vmin=-1, vmax=1.5)
            plt.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(compplot_out_dir, f"test_{i}_{j_plot}_pred_1.pdf"), format='pdf')
            fig.show()

    print(f"Trajectory {i}: mean error = {np.mean(norm_err[i])}, std error = {np.std(norm_err[i])}")