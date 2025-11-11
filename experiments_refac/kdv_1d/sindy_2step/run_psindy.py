from experiments_refac.kdv_1d.sindy_2step.train_psindy import main as train_psindy_main
from experiments_refac.kdv_1d.sindy_2step.postproc_psindy import main as postproc_psindy_main
import pathlib
import sys
import time
import json

OVERWRITE = False
CURR_DIR = str(pathlib.Path(__file__).parent.absolute())

if __name__ == "__main__":
    arg_str = " ".join(sys.argv[1:])
    if len(arg_str) != 0:
        json_str = arg_str.replace("{", '{"').replace(": ", '": ').replace(", ", ', "').replace("pod", '"pod"').replace("ae", '"ae"').replace("data.pkl", '"data.pkl"')
        hparams = json.loads(json_str)
    else:
        hparams = {'data_file': "data_10_5_5",
                'dim_z': 25,
                'threshold': 1e-1,
                'degree': 2,
                }
    print(hparams, flush=True)

    train_psindy_main(**hparams, overwrite=OVERWRITE)
    postproc_psindy_main(**hparams)