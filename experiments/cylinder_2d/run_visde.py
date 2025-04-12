from experiments.cylinder_2d.train_visde import main as train_visde_main
from experiments.cylinder_2d.postproc_visde import main as postproc_visde_main
from experiments.cylinder_2d.plot_kernel import main as plot_kernel_main
from experiments.cylinder_2d.make_animation import main as make_animation_main
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
    #make_animation_main(**hparams)