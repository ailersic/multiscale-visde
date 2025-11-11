from experiments_refac.shallow_water_2d.sindy_2step.train_psindy import main as train_psindy_main
from experiments_refac.shallow_water_2d.sindy_2step.postproc_psindy import main as postproc_psindy_main
import pathlib
import sys
import time
import json

OVERWRITE = False
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

if __name__ == "__main__":
    hparams = {'dim_z': 69,
            'threshold': 1e-2,
            'degree': 1,
            }
    print(hparams, flush=True)

    train_psindy_main(**hparams, overwrite=OVERWRITE)
    postproc_psindy_main(**hparams)