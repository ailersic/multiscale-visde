from dataclasses import dataclass

@dataclass
class CaseConfig:
    data_file: str = "data_20_5_5"
    dim_z_macro: int = 8*8
    dim_z_micro: int = 0
    n_batch: int = 64
    n_sigma: int = 2
    n_win: int = 1
    max_epochs: int = 100
    lr: float = 1e-3
    lr_sched_freq: int = 2000
    augment: bool = True

DEFAULT_CONFIG = CaseConfig()

def get_version_str(config: CaseConfig, suffix: str = None) -> str:
    if config.augment and config.dim_z_micro > 0:
        suffix = "augment" if suffix is None else "_".join(["augment", suffix])
    
    return "_".join([config.data_file,
                     str(config.dim_z_macro),
                     str(config.dim_z_micro),
                     str(config.n_batch),
                     str(config.n_sigma),
                     str(config.n_win),
                     str(config.max_epochs),
                     str(config.lr),
                     str(config.lr_sched_freq)
                     ] + ([] if suffix is None else [suffix]))

if __name__ == "__main__":
    config = DEFAULT_CONFIG
    version = get_version_str(config)
    print(f"Version string: {version}")
    version = get_version_str(config, suffix="test")
    print(f"Version string with suffix: {version}")
    config.augment = True
    version = get_version_str(config)
    print(f"Version string with augmentation: {version}")
    version = get_version_str(config, suffix="test")
    print(f"Version string with augmentation and suffix: {version}")