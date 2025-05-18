from experiments.cylinder_2d.train_visde import main as train_visde_main
from experiments.cylinder_2d.postproc_visde import main as postproc_visde_main
from experiments.cylinder_2d.plot_kernel import main as plot_kernel_main
from experiments.cylinder_2d.comparison_plot import main as comparison_plot_main
from experiments.cylinder_2d.multiscale_plot import main as multiscale_plot_main

import json
import pathlib
import sys

OVERWRITE = False
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

if __name__ == "__main__":
    arg_str = " ".join(sys.argv[1:])
    if len(arg_str) != 0:
        json_str = arg_str.replace("{", '{"').replace(": ", '": ').replace(", ", ', "')
        hparams = json.loads(json_str)
    else:
        hparams = {
            "dim_z_macro": 2*32*8,
            "dim_z_micro": 0,
            "max_epochs": 2000,
            "lr": 1e-3,
            "lr_sched_freq": 2000,
            "augment": True,
        }

    for dim_z_micro in range(6):
        hparams["dim_z_micro"] = dim_z_micro
        if dim_z_micro > 0:
            hparams["lr"] = 1e-4
        print(hparams)

        train_visde_main(**hparams, overwrite=OVERWRITE)
        plot_kernel_main(**hparams)
        postproc_visde_main(**hparams)
        if dim_z_micro == 0 or dim_z_micro == 5:
            comparison_plot_main(**hparams)
        if dim_z_micro == 5:
            multiscale_plot_main(**hparams)