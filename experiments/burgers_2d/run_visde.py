from experiments.burgers_2d.train_visde import main as train_visde_main
from experiments.burgers_2d.postproc_visde import main as postproc_visde_main
from experiments.burgers_2d.plot_kernel import main as plot_kernel_main
from experiments.burgers_2d.make_animation import main as make_animation_main
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
            "dim_z_macro": 64,
            "dim_z_micro": 5,
            "max_epochs": 50,
            "lr": 1e-3,
            "lr_sched_freq": 1000
        }
    print(hparams)

    train_visde_main(**hparams, overwrite=OVERWRITE)
    plot_kernel_main(**hparams)
    postproc_visde_main(**hparams)
    #make_animation_main(**hparams)