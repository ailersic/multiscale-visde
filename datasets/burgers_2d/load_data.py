import os
import pathlib
import pickle as pkl

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
EXT = ".pkl"

def load_data(data_file: str = "data_20_5_5"):
    data_path = os.path.join(CURR_DIR, data_file + EXT)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    
    return data

if __name__ == "__main__":
    data = load_data()
    print("Data loaded successfully.")
    for key, value in data.items():
        print(f"{key}: {value.shape}")
        print(type(value))
        print(value.dtype)