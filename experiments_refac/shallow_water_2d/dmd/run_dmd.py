import numpy as np
import scipy
import scipy.integrate
import pathlib
import os
import pickle as pkl

from matplotlib import animation
from matplotlib import pyplot as plt
from pydmd import DMD
from pydmd.plotter import plot_modes_2D

from datasets.shallow_water_2d.load_data import load_data

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
TRAIN_VAL_TEST = "train"
out_dir = os.path.join(CURR_DIR, "..", "plot_dmd")
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

def train_dmd():
    data = load_data()

    mu = data[f"{TRAIN_VAL_TEST}_mu"].cpu().detach().numpy()
    t = data[f"{TRAIN_VAL_TEST}_t"].cpu().detach().numpy()
    x = data[f"{TRAIN_VAL_TEST}_x"].cpu().detach().numpy()
    f = data[f"{TRAIN_VAL_TEST}_f"].cpu().detach().numpy()

    n_tsteps = t.shape[1]

    snapshots = x[0].reshape(n_tsteps, -1).T  # [n_features, n_tsteps]

    dmd = DMD(svd_rank=69, tikhonov_regularization=0.01)
    dmd.fit(snapshots)

    channel = 0  # Select channel to visualize
    num_modes = 6
    modes = dmd.modes.T  # Shape: [n_modes, n_features]
    print(f"Modes shape: {modes.shape}")

    plt.figure(figsize=(15, 2.5 * num_modes))
    for i in range(min(num_modes, modes.shape[0])):
        mode_vec = modes[i].real  # Take real part
        mode_reshaped = mode_vec.reshape(x.shape[2:])  # Reshape to original spatial dimensions

        plt.subplot(num_modes, 1, i+1)
        plt.imshow(mode_reshaped[channel])
        plt.title(f"DMD Mode {i+1} (Channel {channel})")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(CURR_DIR, "dmd_modes.png"), dpi=300)

    return dmd

def pred_dmd(dmd: DMD):
    data = load_data()

    mu = data["test_mu"].cpu().detach().numpy()
    t = data["test_t"].cpu().detach().numpy()
    x = data["test_x"].cpu().detach().numpy()
    f = data["test_f"].cpu().detach().numpy()

    n_traj = t.shape[0]
    n_tsteps = t.shape[1]

    # Predicting future states using DMD
    norm_err = np.zeros((n_traj, n_tsteps))
    tsamples = [0, n_tsteps//4, n_tsteps//2, 3*n_tsteps//4, n_tsteps-1]
    cmap = plt.get_cmap("ocean")

    for i in range(n_traj):
        x_i = x[i].reshape(n_tsteps, -1)
        x_i_0 = x_i[0]  # Initial condition
        x_i_dmd = np.zeros_like(x_i)
        x_i_dmd[0] = x_i_0

        for j in range(1, x_i.shape[0]):
            print(f"Predicting step {j}/{x_i.shape[0]-1}")
            x_i_dmd[j] = dmd.predict(x_i_dmd[j-1]).real

        if i == 0:
            for j_plot, j_t in enumerate(tsamples):
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(x_i_dmd[j_t].reshape(x.shape[2:])[0], cmap=cmap, vmin=-1, vmax=1)
                plt.axis("off")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"test_{i}_{j_plot}_pred.pdf"), format='pdf')
                fig.show()

        for j in range(x_i.shape[0]):
            norm_err[i, j] = np.linalg.norm(x_i_dmd[j] - x_i[j]) / np.linalg.norm(x_i[j])

        print(f"Mean error = {np.mean(norm_err[i])}, std error = {np.std(norm_err[i])}")

        print(f"Predicted future states shape: {x_i_dmd.shape}")

        # Visualizing the predicted states
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(x_i[0].reshape(x.shape[2:])[0], label="Original")
        axs[1].imshow(x_i_dmd[0].reshape(x.shape[2:])[0], label="Predicted")
        axs[0].set_title("Original State")
        axs[1].set_title("Predicted State using DMD")
        plt.tight_layout()
        plt.savefig(os.path.join(CURR_DIR, f"dmd_prediction_0_{i}.png"), dpi=300)
        plt.show()

        # Visualizing the predicted states at final time step
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(x_i[-1].reshape(x.shape[2:])[0], label="Original")
        axs[1].imshow(x_i_dmd[-1].reshape(x.shape[2:])[0], label="Predicted")
        axs[0].set_title("Original Final State")
        axs[1].set_title("Predicted Final State using DMD")
        plt.tight_layout()
        plt.savefig(os.path.join(CURR_DIR, f"dmd_prediction_final_{i}.png"), dpi=300)
        plt.show()

    print(f"Mean error = {np.mean(norm_err)}, std error = {np.std(norm_err)}")

    np.set_printoptions(threshold=np.inf)
    with open(os.path.join(CURR_DIR, "test_error.txt"), "w") as file:
        file.write(np.array2string(norm_err, precision=5))
        file.write("\n")
        file.write(f"Mean: {np.mean(norm_err):.5f}, Std Dev: {np.std(norm_err):.5f}\n")

if __name__ == "__main__":
    dmd_model = train_dmd()
    pred_dmd(dmd_model)