from experiments.kdv_1d.train_visde import main as train_visde_main
from experiments.kdv_1d.postproc_visde import main as postproc_visde_main
from experiments.kdv_1d.plot_kernel import main as plot_kernel_main
#from experiments.kdv_1d.multiscale_plot import main as multiscale_plot_main

import json
import pathlib
import sys

OVERWRITE = False
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

if __name__ == "__main__":
    arg_str = " ".join(sys.argv[1:])
    json_str = arg_str.replace("{", '{"').replace(": ", '": ').replace(", ", ', "')
    hparams = json.loads(json_str)
    print(hparams)

    train_visde_main(**hparams, overwrite=OVERWRITE)
    plot_kernel_main(**hparams)
    postproc_visde_main(**hparams)
    #multiscale_plot_main(**hparams)