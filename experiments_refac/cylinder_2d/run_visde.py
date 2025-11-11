from experiments_refac.cylinder_2d.train_visde import main as train_visde_main
from experiments_refac.cylinder_2d.postproc_visde import main as postproc_visde_main
from experiments_refac.cylinder_2d.plot_kernel import main as plot_kernel_main
from experiments_refac.cylinder_2d.utils import CaseConfig, DEFAULT_CONFIG, get_version_str

import pathlib
import sys

OVERWRITE = False
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

if __name__ == "__main__":
    config = DEFAULT_CONFIG

    for dim_z_micro in range(0, 6):
        config.dim_z_micro = dim_z_micro
        if dim_z_micro > 0:
            config.lr = 1e-4
        print(config)

        train_visde_main(config, overwrite=OVERWRITE)
        plot_kernel_main(config)
        postproc_visde_main(config)