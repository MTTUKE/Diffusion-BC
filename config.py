from dataclasses import dataclass

@dataclass
class Config:
    data_root: str = "data"
    train_imgs_dir: str = "193_train/193(region_growth)/imgs"
    train_masks_dir: str = "193_train/193(region_growth)/masks"
    test_imgs_dir: str = "193_test/imgs"
    test_masks_dir: str = "193_test/masks(region)"
    img_size: int = 256

    val_fraction: float = 0.10
    split_seed: int = 42

    batch_size: int = 6
    num_workers: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    use_amp: bool = True

    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.05

    schedule_type: str = "cosine"   # "linear" or "cosine"
    cosine_s: float = 0.008

    base_channels: int = 64
    dropout: float = 0.1
    ema_decay: float = 0.9995

    sample_val_every_epochs: int = 2
    sample_val_images: int = 32
    val_ddim_steps: int = 120
    val_n_samples: int = 4

    early_stop_patience: int = 4
    early_stop_min_delta: float = 0.005

    thr_grid_start: float = 0.10
    thr_grid_end: float = 0.90
    thr_grid_step: float = 0.02

    ckpt_dir: str = "checkpoints"
    log_dir: str = "runs"
    save_every_epochs: int = 2