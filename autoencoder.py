import argparse
from pathlib import Path

import pytorch_lightning as L
import torch
from cleandiffuser.dataset.libero_dataset import LiberoDataset
from cleandiffuser.utils import set_seed
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import AutoEncoder, DatasetWrapper, VariationalAutoEncoder

DATASET_PATH = "/home/dzb/CleanDiffuser/dev/libero"
SAVE_PATH = "/home/dzb/cocos/results"

argparser = argparse.ArgumentParser()
argparser.add_argument("--mode", type=str, default="training")
argparser.add_argument("--task_suite", type=str, default="libero_goal")
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--devices", type=int, nargs="+", default=[0])
argparser.add_argument("--task_id", type=int, default=0)
argparser.add_argument("--t5_path", type=str, default="google-t5/t5-base")
argparser.add_argument("--vit_path", type=str, default="facebook/dinov2-base")
argparser.add_argument("--model_type", type=str, default="autoencoder", choices=["autoencoder", "vae"])
argparser.add_argument("--training_steps", type=int, default=10_000)
args = argparser.parse_args()

task_suite = args.task_suite
task_id = args.task_id
t5_pretrained_model_name_or_path = args.t5_path
vit_pretrained_model_name_or_path = args.vit_path
seed = args.seed
mode = args.mode
devices = args.devices
model_type = args.model_type
training_steps = args.training_steps

dataset_path = Path(DATASET_PATH) / f"{task_suite}.zarr"
default_root_dir = Path(SAVE_PATH) / model_type / task_suite


if __name__ == "__main__":
    set_seed(seed)

    # --- Dataset ---
    dataset = LiberoDataset(data_path=dataset_path, observation_meta=["color", "color_ego"], To=1, Ta=1)

    dataloader = torch.utils.data.DataLoader(
        DatasetWrapper(dataset),
        batch_size=256 // len(devices),
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )

    if model_type == "autoencoder":
        autoencoder = AutoEncoder(
            vit_path=vit_pretrained_model_name_or_path,
            t5_path=t5_pretrained_model_name_or_path,
            img_len=257 * 2,
            lang_len=32,
            z_dim=7,
            z_len=16,
            hidden_size=384,
            nheads=6,
        )
    elif model_type == "vae":
        autoencoder = VariationalAutoEncoder(
            vit_path=vit_pretrained_model_name_or_path,
            t5_path=t5_pretrained_model_name_or_path,
            img_len=257 * 2,
            lang_len=32,
            z_dim=7,
            z_len=16,
            hidden_size=384,
            nheads=6,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # -- Training ---
    if mode == "training":
        callback = ModelCheckpoint(
            dirpath=default_root_dir,
            every_n_train_steps=5000,
            save_top_k=-1,
            filename="{step}",
        )
        trainer = L.Trainer(
            devices=devices,
            max_steps=training_steps,
            callbacks=[callback],
            default_root_dir=default_root_dir,
            precision="bf16-mixed",
            strategy="auto",
            accumulate_grad_batches=1,
        )
        trainer.fit(autoencoder, dataloader)
