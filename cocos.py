import argparse
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import einops
import gym
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision.transforms as T
from cleandiffuser.dataset.libero_dataset import LiberoDataset
from cleandiffuser.diffusion import ContinuousRectifiedFlow
from cleandiffuser.env import libero
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import DiT1dWithACICrossAttention
from cleandiffuser.utils import UntrainablePositionalEmbedding, set_seed
from pytorch_lightning.callbacks import ModelCheckpoint
from termcolor import cprint
from transformers import AutoModel, T5Config

from utils import AutoEncoder, DatasetWrapper, T5LanguageEncoder

DATASET_PATH = "/home/dzb/CleanDiffuser/dev/libero"
SAVE_PATH = "/home/dzb/cocos/results"


class ViTAndT5VisionLanguageCondition(BaseNNCondition):
    def __init__(
        self,
        autoencoder_path: Optional[str],
        vit_path: str = "facebook/dinov2-base",
        t5_hidden_dim: int = 768,
        emb_dim: int = 384,
        freeze: bool = True,
        To: int = 1,
        n_views: int = 2,
        ema_update_autoencoder: bool = False,
    ):
        super().__init__()
        self.vit_model = AutoModel.from_pretrained(vit_path)
        if freeze:
            for p in self.vit_model.parameters():
                p.requires_grad = False

        vit_hidden_dim = self.vit_model.config.hidden_size
        self.num_hidden_layers = self.vit_model.config.num_hidden_layers

        self.vit_adapter = nn.Sequential(nn.Linear(vit_hidden_dim, emb_dim), nn.GELU(approximate="tanh"))
        self.t5_adapter = nn.Sequential(nn.Linear(t5_hidden_dim, emb_dim), nn.GELU(approximate="tanh"))
        self.To_pos_emb = nn.Parameter(torch.randn(1, To * n_views, 1, emb_dim) * 0.02)
        num_patches = (224 // self.vit_model.config.patch_size) ** 2

        vit_pos_emb = UntrainablePositionalEmbedding(emb_dim, 1000)(torch.arange(num_patches))
        vit_pos_emb = vit_pos_emb[None, None] * 0.2
        self.vit_pos_emb = nn.Parameter(vit_pos_emb)

        t5_pos_emb = UntrainablePositionalEmbedding(emb_dim, 100)(torch.arange(32))
        t5_pos_emb = t5_pos_emb[None] * 0.2
        self.t5_pos_emb = nn.Parameter(t5_pos_emb)

        autoencoder = AutoEncoder(vit_path=vit_path, img_len=257 * 2)
        if autoencoder_path is not None:
            ckpt = torch.load(autoencoder_path, map_location="cpu", weights_only=True)
            ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            autoencoder.load_state_dict(ckpt, strict=False)
        if ema_update_autoencoder:
            self.autoencoder_ema = deepcopy(self.autoencoder).eval()
            for p in self.autoencoder_ema.parameters():
                p.requires_grad = False
        else:
            self.autoencoder_ema = None
            self.autoencoder = autoencoder.eval()
            for p in self.autoencoder.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def ema_update_ae(self):
        if self.autoencoder_ema is not None:
            for p, p_ema in zip(self.autoencoder.parameters(), self.autoencoder_ema.parameters()):
                p_ema.copy_(p_ema * 0.999 + p * (1 - 0.999))

    def forward(
        self,
        condition: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        requires_attn_map: bool = False,
    ):
        image = condition["image"]
        language_embedding = condition["language_embedding"]
        language_mask = condition["language_mask"]
        b = image.shape[0]

        # transform to pytorch attention padding mask, where True means padding.
        if language_mask is not None:
            language_mask = torch.logical_not(language_mask.to(torch.bool))

        image = einops.rearrange(image, "b To C H W -> (b To) C H W")
        vision_output = self.vit_model(image, output_attentions=requires_attn_map)
        vision_feat = vision_output["last_hidden_state"][:, 1:]  # (b * To, 196, 768)
        vision_feat = self.vit_adapter(vision_feat)
        vision_feat = einops.rearrange(vision_feat, "(b To) m d -> b To m d", b=b)
        vision_feat = vision_feat + self.To_pos_emb + self.vit_pos_emb
        vision_feat = einops.rearrange(vision_feat, "b To m d -> b (To m) d")

        with torch.no_grad():
            autoencoder = self.autoencoder_ema if self.autoencoder_ema else self.autoencoder
            img_src = vision_output["last_hidden_state"]
            img_src = einops.rearrange(img_src, "(b To) m d -> b (To m) d", b=b)
            img_mask = torch.zeros(
                (b, autoencoder.img_len),
                device=language_mask.device,
                dtype=language_mask.dtype,
            )
            mask = torch.cat([img_mask, language_mask], dim=1)
            x1 = autoencoder.encode(img_src, language_embedding, mask)
            x1 = x1 + (torch.randn_like(x1) * 0.2).clip(-0.4, 0.4)

        lang_feat = self.t5_adapter(language_embedding) + self.t5_pos_emb

        self.ema_update_ae()

        return {
            "vec_condition": None,
            "lang_condition": lang_feat,
            "lang_condition_mask": language_mask,
            "vis_condition": vision_feat,
            "vis_condition_mask": None,
            "x1": x1,
        }


class ConditionRectifiedFlow(ContinuousRectifiedFlow):
    def loss(
        self,
        x0: torch.Tensor,
        condition: torch.Tensor = None,
        x1: torch.Tensor = None,
    ):
        condition = self.model["condition"](condition) if condition is not None else None
        x1 = condition["x1"] if x1 is None else x1

        xt, t, _ = self.add_noise(x0, eps=x1)

        loss = (self.model["diffusion"](xt, t, condition) - (x0 - x1)) ** 2

        return (loss * self.loss_weight * (1 - self.fix_mask)).mean()


argparser = argparse.ArgumentParser()
argparser.add_argument("--mode", type=str, default="inference")
argparser.add_argument("--task_suite", type=str, default="libero_goal")
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--devices", type=int, nargs="+", default=[0])
argparser.add_argument("--task_id", type=int, default=0)
argparser.add_argument("--t5_path", type=str, default="google-t5/t5-base")
argparser.add_argument("--vit_path", type=str, default="facebook/dinov2-base")
argparser.add_argument("--model_type", type=str, default="autoencoder", choices=["autoencoder", "vae"])
argparser.add_argument("--training_steps", type=int, default=30_000)
argparser.add_argument("--ckpt_file", type=str, default="step=30000.ckpt")
argparser.add_argument("--ema_update_autoencoder", type=bool, default=False)
args = argparser.parse_args()


task_suite = args.task_suite
t5_pretrained_model_name_or_path = args.t5_path
vit_pretrained_model_name_or_path = args.vit_path
seed = args.seed
mode = args.mode
devices = args.devices
task_id = args.task_id
model_type = args.model_type
training_steps = args.training_steps
ckpt_file = args.ckpt_file
ema_update_autoencoder = args.ema_update_autoencoder

dataset_path = Path(DATASET_PATH) / f"{task_suite}.zarr"
default_root_dir = Path(SAVE_PATH) / "cocos" / task_suite
autoencoder_path = Path(SAVE_PATH) / model_type / task_suite / "step=10000.ckpt"
env_name = task_suite.replace("_", "-") + "-v0"

# Do not change if you want to use pre-trained ckpt
To = 1
Ta = 16
num_act_exec = 8
sampling_steps = 20
NORM_PARAMS = (0.5, 0.5, 0.5)
max_episode_steps = 600 if "10" in task_suite else 300


if __name__ == "__main__":
    set_seed(seed)

    # --- Dataset ---
    dataset = LiberoDataset(data_path=dataset_path, observation_meta=["color", "color_ego"], To=To, Ta=Ta)
    act_dim = 7

    dataloader = torch.utils.data.DataLoader(
        DatasetWrapper(dataset),
        batch_size=256 // len(devices),
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )

    # --- Model ---
    t5_hidden_dim = T5Config.from_pretrained(t5_pretrained_model_name_or_path).d_model

    nn_diffusion = DiT1dWithACICrossAttention(
        x_dim=act_dim,
        x_seq_len=Ta,
        emb_dim=768,
        d_model=384,
        n_heads=6,
        depth=12,
        timestep_emb_type="untrainable_fourier",
        timestep_emb_params={"scale": 0.2},
    )
    nn_condition = ViTAndT5VisionLanguageCondition(
        autoencoder_path=autoencoder_path,
        vit_path=vit_pretrained_model_name_or_path,
        t5_hidden_dim=t5_hidden_dim,
        emb_dim=768,
        freeze=True,
        To=To,
        n_views=2,
        ema_update_autoencoder=ema_update_autoencoder,
    )

    policy = ConditionRectifiedFlow(
        nn_diffusion=nn_diffusion,
        nn_condition=nn_condition,
        x_max=torch.full((Ta, act_dim), 1.0),
        x_min=torch.full((Ta, act_dim), -1.0),
    )

    # -- Training ---
    if mode == "training":
        callback = ModelCheckpoint(
            dirpath=default_root_dir,
            every_n_train_steps=20,
            save_top_k=-1,
            filename="{step}",
        )
        trainer = L.Trainer(
            devices=devices,
            max_steps=training_steps,
            callbacks=[callback],
            default_root_dir=default_root_dir,
            precision="bf16-mixed",
            strategy="ddp_find_unused_parameters_true",
            accumulate_grad_batches=1,
        )
        trainer.fit(policy, dataloader)

    # -- Inference --
    elif mode == "inference":
        device = f"cuda:{devices[0]}"
        t5_language_encoder = T5LanguageEncoder(
            pretrained_model_name_or_path=t5_pretrained_model_name_or_path,
            max_length=32,
            device=f"cuda:{devices[0]}",
        )
        nn_condition.autoencoder = nn_condition.autoencoder.to(device).eval()

        env = gym.make(
            env_name,
            task_id=task_id,
            image_size=224,
            require_depth=False,
            require_point_cloud=False,
            seed=seed,
            max_episode_steps=max_episode_steps,
        )
        cprint(f"TASK: {env.task_description}", "green", attrs=["bold"])

        lang_emb, lang_mask = t5_language_encoder([env.task_description])
        bool_lang_mask = torch.logical_not(lang_mask.to(torch.bool))
        img_mask = torch.zeros(
            (1, nn_condition.autoencoder.img_len),
            device=bool_lang_mask.device,
            dtype=bool_lang_mask.dtype,
        )
        mask = torch.cat([img_mask, bool_lang_mask], dim=1)

        normalizer = dataset.get_normalizer()
        img_norm = T.Normalize(NORM_PARAMS, NORM_PARAMS)
        center_crop = T.Compose([T.Normalize(NORM_PARAMS, NORM_PARAMS), T.CenterCrop(200), T.Resize(224)])
        resize = T.Resize((224, 224))

        ckpt = torch.load(default_root_dir / ckpt_file, map_location="cpu", weights_only=True)
        ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        policy.load_state_dict(ckpt, strict=False)
        policy = policy.to(device).eval()

        prior = torch.zeros((1, Ta, act_dim), device=device)

        success = []

        def process_obs(obs):
            image = obs["agentview_image"] / 255.0
            image = torch.tensor(image, device=device, dtype=torch.float32)
            image = center_crop(image)
            image = image[None, None].repeat(1, To, 1, 1, 1)
            image_ego = obs["robot0_eye_in_hand_image"] / 255.0
            image_ego = torch.tensor(image_ego, device=device, dtype=torch.float32)
            image_ego = img_norm(image_ego)
            image_ego = image_ego[None, None].repeat(1, To, 1, 1, 1)
            dinov2_input = torch.cat([image, image_ego], 1)[0]
            img_src = nn_condition.autoencoder.dinov2(dinov2_input)
            img_src = einops.rearrange(img_src, "(b t) n d -> b (t n) d", b=1)
            return image, image_ego, img_src

        for init_state_id in range(env.num_init_states):
            dummy_steps = 0
            obs, all_done, all_rew = env.reset(init_state_id=init_state_id), False, 0

            image, image_ego, img_src = process_obs(obs)
            with torch.no_grad():
                x1 = nn_condition.autoencoder.encode(img_src, lang_emb, mask)

            x1 = x1 + (torch.randn_like(x1) * 0.2).clip(-0.4, 0.4)
            image = torch.cat([image, image_ego], dim=1)

            while not np.all(all_done):
                act, log = policy.sample(
                    prior,
                    x1=x1,
                    solver="euler",
                    sample_steps=sampling_steps,
                    condition_cfg={
                        "image": image,
                        "language_embedding": lang_emb,
                        "language_mask": lang_mask,
                    },
                    use_ema=False,
                    w_cfg=1.0,
                )
                act = normalizer["action"].unnormalize(act.cpu().numpy())

                # Objects may fall down from sky when initializing the environment
                # Take a few dummy steps to stabilize the environment
                if dummy_steps < 2:
                    act = np.zeros((1, num_act_exec, act_dim), dtype=np.float32)
                    dummy_steps += 1

                for i in range(num_act_exec):
                    next_obs, rew, done, _ = env.step(act[0, i])
                    all_done = np.logical_or(all_done, done)
                    all_rew += rew

                    if i >= num_act_exec - To:
                        this_image, this_image_ego, img_src = process_obs(next_obs)
                        with torch.no_grad():
                            x1 = nn_condition.autoencoder.encode(img_src, lang_emb, mask)
                        x1 = x1 + (torch.randn_like(x1) * 0.2).clip(-0.4, 0.4)
                        image[:, i - num_act_exec + To] = this_image[:, 0]
                        image[:, i - num_act_exec + To * 2] = this_image_ego[:, 0]

                    if np.all(all_done):
                        break

                cprint(f"[Test {init_state_id}] Success: {all_rew}", "green")

            if all_rew:
                success.append(True)

        print(f"Success rate: {np.sum(success) / env.num_init_states}")
