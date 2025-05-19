from typing import List, Optional

import einops
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision.transforms as T
from cleandiffuser.utils import UntrainablePositionalEmbedding
from transformers import AutoConfig, AutoModel


class RepresentationModel:
    def __init__(
        self,
        model_path: str = "facebook/dinov2-with-registers-base",
        device: str = "cpu",
    ):
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self.device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device:
            self.model = self.model.to(x.device)
            self.device = x.device

        with torch.no_grad():
            return self.model(x).last_hidden_state


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 384,
        nheads: int = 6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, nheads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.LayerNorm(4 * hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attention(h, h, h, key_padding_mask=mask)[0]
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        src_dim: int = 768,
        src_len: int = 261,
        tgt_dim: int = 7,
        tgt_len: int = 16,
        hidden_size: int = 384,
        nheads: int = 6,
        add_tanh: bool = True,
    ):
        super().__init__()
        self.tgt_len = tgt_len
        self.adapter = nn.Linear(src_dim, hidden_size)

        self.tgt_emb = nn.Parameter(torch.randn((1, 1, hidden_size)) * 0.02)

        pos_indices = torch.arange(tgt_len + src_len).unsqueeze(0)
        pos_emb = UntrainablePositionalEmbedding(hidden_size, max_positions=1000)(pos_indices)
        self.pos_emb = nn.Parameter(pos_emb * 0.2)

        self.transformer = TransformerLayer(hidden_size, nheads)

        self.out_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, tgt_dim),
            nn.Tanh() if add_tanh else nn.Identity(),
        )

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        b = src.size(0)

        if mask is not None:
            # `False` means not padding
            tgt_mask = torch.zeros((b, self.tgt_len), device=src.device, dtype=torch.bool)
            mask = torch.cat([tgt_mask, mask], dim=1)

        tgt_emb = self.tgt_emb.expand(b, self.tgt_len, -1)
        src_emb = self.adapter(src)
        x = torch.cat([tgt_emb, src_emb], dim=1)
        x = x + self.pos_emb
        x = self.transformer(x, mask)
        return self.out_layer(x[:, : self.tgt_len])


class Decoder(nn.Module):
    def __init__(
        self,
        src_dim: int = 768,
        src_len: int = 261,
        tgt_dim: int = 7,
        tgt_len: int = 16,
        hidden_size: int = 384,
        nheads: int = 6,
    ):
        super().__init__()
        self.tgt_len = tgt_len
        self.src_len = src_len
        self.adapter = nn.Linear(tgt_dim, hidden_size)

        self.src_emb = nn.Parameter(torch.randn((1, 1, hidden_size)) * 0.02)

        pos_indices = torch.arange(tgt_len + src_len).unsqueeze(0)
        pos_emb = UntrainablePositionalEmbedding(hidden_size, max_positions=1000)(pos_indices)
        self.pos_emb = nn.Parameter(pos_emb * 0.2)

        self.transformer = TransformerLayer(hidden_size, nheads)

        self.out_layer = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, src_dim))

    def forward(self, tgt: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        b = tgt.size(0)

        if mask is not None:
            # `False` means not padding
            tgt_mask = torch.zeros((b, self.tgt_len), device=tgt.device, dtype=torch.bool)
            mask = torch.cat([tgt_mask, mask], dim=1)

        src_emb = self.src_emb.expand(b, self.src_len, -1)
        tgt_emb = self.adapter(tgt)
        x = torch.cat([tgt_emb, src_emb], dim=1)
        x = x + self.pos_emb
        x = self.transformer(x, mask)
        return self.out_layer(x[:, self.tgt_len :])


class AutoEncoder(L.LightningModule):
    """
    ==================== Encoder ====================
    2-view RGB (b, 2, 3, 224, 224) -> |DINOv2| -> (b, 2*257, 768) -> |         |
                                                                     | Encoder | -> (b, horizon, act_dim)
    language (b, 32)               ->   |T5|   -> (b,    32, 768) -> |         |
    =================================================

    ===================== Decoder ====================
                             |         | -> (b, 2*257, 768)
    (b, horizon, act_dim) -> | Decoder |
                             |         | -> (b,    32, 768)
    ==================================================

    """

    def __init__(
        self,
        vit_path: str = "facebook/dinov2-base",
        t5_path: str = "google-t5/t5-base",
        img_len: int = 257 * 2,
        lang_len: int = 32,
        z_dim: int = 7,
        z_len: int = 16,
        hidden_size: int = 384,
        nheads: int = 6,
    ):
        super().__init__()
        vit_config = AutoConfig.from_pretrained(vit_path)
        t5_config = AutoConfig.from_pretrained(t5_path)
        img_dim = vit_config.hidden_size
        lang_dim = t5_config.d_model
        self.lang_len = lang_len
        self.img_len = img_len
        self.tgt_len = z_len

        self.dinov2 = RepresentationModel(vit_path)

        # print(img_dim, img_len, lang_dim, lang_len)

        self.img_encoder = Encoder(img_dim, img_len, z_dim, int(z_len * 0.75), hidden_size, nheads)
        self.img_decoder = Decoder(img_dim, img_len, z_dim, int(z_len * 0.75), hidden_size, nheads)
        self.lang_encoder = Encoder(lang_dim, lang_len, z_dim, int(z_len * 0.25), hidden_size, nheads)
        self.lang_decoder = Decoder(lang_dim, lang_len, z_dim, int(z_len * 0.25), hidden_size, nheads)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def encode_with_raw_image(
        self, img: torch.Tensor, lang_emb: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        img_emb = self.dinov2(img)
        return self.encode(img_emb, lang_emb, mask), img_emb

    def encode(
        self,
        img_emb: torch.Tensor,
        lang_emb: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert mask.shape[1] == (self.img_len + self.lang_len)
        img_tgt = self.img_encoder(img_emb, mask[:, : self.img_len])
        lang_tgt = self.lang_encoder(lang_emb, mask[:, self.img_len :])
        return torch.cat([img_tgt, lang_tgt], dim=1)

    def decode(self, z: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        assert mask.shape[1] == (self.img_len + self.lang_len)
        img_emb = self.img_decoder(z[:, : int(self.tgt_len * 0.75)], mask[:, : self.img_len])
        lang_emb = self.lang_decoder(z[:, int(self.tgt_len * 0.75) :], mask[:, self.img_len :])
        return img_emb, lang_emb

    def loss(
        self,
        img_emb: torch.Tensor,
        lang_emb: torch.Tensor,
        lang_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b = img_emb.shape[0]
        img_mask = torch.zeros((b, self.img_len), device=lang_mask.device, dtype=lang_mask.dtype)
        mask = torch.cat([img_mask, lang_mask], dim=1)
        z = self.encode(img_emb, lang_emb, mask)
        recon_img_emb, recon_lang_emb = self.decode(z, mask)

        img_loss = nn.functional.cosine_similarity(recon_img_emb, img_emb, dim=-1)
        lang_loss = nn.functional.cosine_similarity(recon_lang_emb, lang_emb, dim=-1)
        if lang_mask is not None:
            loss_mask = torch.logical_not(lang_mask).to(lang_emb.dtype)
            lang_loss = -(lang_loss * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            lang_loss = -lang_loss.mean(-1)
        img_loss = -img_loss.mean(-1)

        return img_loss, lang_loss, 1.0, 0.1

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.encoder(src, mask)

    def training_step(self, batch, batch_idx):
        condition = batch["condition_cfg"]
        image = condition["image"]
        b = image.shape[0]
        image = einops.rearrange(image, "b t c h w -> (b t) c h w")
        img_src = self.dinov2(image)
        img_src = einops.rearrange(img_src, "(b t) n d -> b (t n) d", b=b)
        lang_src = condition["language_embedding"]
        lang_mask = condition["language_mask"]

        # transform to pytorch attention padding mask, where True means padding.
        if lang_mask is not None:
            lang_mask = torch.logical_not(lang_mask.to(torch.bool))

        img_loss, lang_loss, img_loss_weight, lang_loss_weight = self.loss(img_src, lang_src, lang_mask)
        loss = (img_loss * img_loss_weight + lang_loss * lang_loss_weight).mean()
        self.log("img_loss", img_loss.mean())
        self.log("lang_loss", lang_loss.mean())
        return loss


class VariationalAutoEncoder(L.LightningModule):
    def __init__(
        self,
        vit_path: str = "facebook/dinov2-with-registers-base",
        t5_path: str = "google-t5/t5-base",
        img_len: int = 261 * 2,
        lang_len: int = 32,
        z_dim: int = 7,
        z_len: int = 16,
        hidden_size: int = 384,
        nheads: int = 6,
    ):
        super().__init__()
        vit_config = AutoConfig.from_pretrained(vit_path)
        t5_config = AutoConfig.from_pretrained(t5_path)
        img_dim = vit_config.hidden_size
        lang_dim = t5_config.d_model
        self.lang_len = lang_len
        self.img_len = img_len
        self.tgt_len = z_len

        self.dinov2 = RepresentationModel(vit_path)

        self.img_encoder = Encoder(img_dim, img_len, z_dim * 2, int(z_len * 0.75), hidden_size, nheads, False)
        self.img_decoder = Decoder(img_dim, img_len, z_dim, int(z_len * 0.75), hidden_size, nheads)
        self.lang_encoder = Encoder(lang_dim, lang_len, z_dim * 2, int(z_len * 0.25), hidden_size, nheads, False)
        self.lang_decoder = Decoder(lang_dim, lang_len, z_dim, int(z_len * 0.25), hidden_size, nheads)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def encode_with_raw_image(
        self, img: torch.Tensor, lang_emb: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        img_emb = self.dinov2(img)
        return self.encode(img_emb, lang_emb, mask), img_emb

    def encode(
        self,
        img_emb: torch.Tensor,
        lang_emb: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert mask.shape[1] == (self.img_len + self.lang_len)
        img_tgt_out = self.img_encoder(img_emb, mask[:, : self.img_len])
        img_tgt_mean, img_tgt_logstd = torch.chunk(img_tgt_out, 2, dim=-1)
        img_tgt_mean = torch.tanh(img_tgt_mean)
        img_tgt_logstd = img_tgt_logstd.clamp(-20.0, 2.0)
        lang_tgt_out = self.lang_encoder(lang_emb, mask[:, self.img_len :])
        lang_tgt_mean, lang_tgt_logstd = torch.chunk(lang_tgt_out, 2, dim=-1)
        lang_tgt_mean = torch.tanh(lang_tgt_mean)
        lang_tgt_logstd = lang_tgt_logstd.clamp(-20.0, 2.0)

        img_tgt = img_tgt_mean + torch.exp(img_tgt_logstd) * torch.randn_like(img_tgt_mean)
        lang_tgt = lang_tgt_mean + torch.exp(lang_tgt_logstd) * torch.randn_like(lang_tgt_mean)
        return torch.cat([img_tgt, lang_tgt], dim=1)

    def decode(self, z: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        assert mask.shape[1] == (self.img_len + self.lang_len)
        img_emb = self.img_decoder(z[:, : int(self.tgt_len * 0.75)], mask[:, : self.img_len])
        lang_emb = self.lang_decoder(z[:, int(self.tgt_len * 0.75) :], mask[:, self.img_len :])
        return img_emb, lang_emb

    def loss(
        self,
        img_emb: torch.Tensor,
        lang_emb: torch.Tensor,
        lang_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b = img_emb.shape[0]
        img_mask = torch.zeros((b, self.img_len), device=lang_mask.device, dtype=lang_mask.dtype)

        img_tgt_out = self.img_encoder(img_emb, img_mask)
        img_tgt_mean, img_tgt_logstd = torch.chunk(img_tgt_out, 2, dim=-1)
        img_tgt_mean = torch.tanh(img_tgt_mean)
        img_tgt_logstd = img_tgt_logstd.clamp(-20.0, 2.0)

        img_tgt = img_tgt_mean + torch.exp(img_tgt_logstd) * torch.randn_like(img_tgt_mean)

        lang_tgt_out = self.lang_encoder(lang_emb, lang_mask)
        lang_tgt_mean, lang_tgt_logstd = torch.chunk(lang_tgt_out, 2, dim=-1)
        lang_tgt_mean = torch.tanh(lang_tgt_mean)
        lang_tgt_logstd = lang_tgt_logstd.clamp(-20.0, 2.0)

        lang_tgt = lang_tgt_mean + torch.exp(lang_tgt_logstd) * torch.randn_like(lang_tgt_mean)

        pred_img_emb = self.img_decoder(img_tgt, img_mask)
        pred_lang_emb = self.lang_decoder(lang_tgt, lang_mask)

        # kl loss
        img_kl_loss = -0.5 * (1 + 2 * img_tgt_logstd - img_tgt_mean.pow(2) - (2 * img_tgt_logstd).exp()).sum(-1)
        lang_kl_loss = -0.5 * (1 + 2 * lang_tgt_logstd - lang_tgt_mean.pow(2) - (2 * lang_tgt_logstd).exp()).sum(-1)

        img_recon_loss = -nn.functional.cosine_similarity(pred_img_emb, img_emb, dim=-1)
        lang_recon_loss = -nn.functional.cosine_similarity(pred_lang_emb, lang_emb, dim=-1)

        if lang_mask is not None:
            loss_mask = torch.logical_not(lang_mask).to(lang_emb.dtype)
            lang_loss = ((lang_recon_loss * loss_mask).sum(-1) / loss_mask.sum(-1)).mean() + 1e-5 * lang_kl_loss.mean()
        else:
            lang_loss = lang_recon_loss.mean()
        img_loss = img_recon_loss.mean() + 0.001 * img_kl_loss.mean()

        self.log("img_kl_loss", img_kl_loss.mean())
        self.log("img_recon_loss", img_recon_loss.mean())
        self.log("lang_kl_loss", lang_kl_loss.mean())
        self.log("lang_recon_loss", lang_recon_loss.mean())

        return img_loss, lang_loss, 1.0, 1.0

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.encoder(src, mask)

    def training_step(self, batch, batch_idx):
        condition = batch["condition_cfg"]
        image = condition["image"]
        b = image.shape[0]
        image = einops.rearrange(image, "b t c h w -> (b t) c h w")
        img_src = self.dinov2(image)
        img_src = einops.rearrange(img_src, "(b t) n d -> b (t n) d", b=b)
        lang_src = condition["language_embedding"]
        lang_mask = condition["language_mask"]

        # transform to pytorch attention padding mask, where True means padding.
        if lang_mask is not None:
            lang_mask = torch.logical_not(lang_mask.to(torch.bool))

        img_loss, lang_loss, img_loss_weight, lang_loss_weight = self.loss(img_src, lang_src, lang_mask)
        loss = (img_loss * img_loss_weight + lang_loss * lang_loss_weight).mean()
        self.log("img_loss", img_loss.mean())
        self.log("lang_loss", lang_loss.mean())
        return loss


class DatasetWrapper:
    def __init__(self, dataset, no_ego=False):
        NORM_PARAMS = (0.5, 0.5, 0.5)
        self.dataset = dataset
        self.no_ego = no_ego
        # add normalization and random crop to 3rd person view
        # add only normalization to egocentric view
        self.normalize = T.Normalize(NORM_PARAMS, NORM_PARAMS)
        self.vis_aug = T.Compose([T.Normalize(NORM_PARAMS, NORM_PARAMS), T.RandomCrop(200), T.Resize(224)])

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Rescale to [0, 1] and apply random crop and resize
        image = item["observation"]["color"].astype(np.float32) / 255.0
        image = self.vis_aug(torch.tensor(image))

        # Use hand-view image
        if "color_ego" in item["observation"].keys():
            image_ego = item["observation"]["color_ego"].astype(np.float32) / 255.0
            image_ego = self.normalize(torch.tensor(image_ego))
            combined_image = torch.cat([image, image_ego], dim=0)
        else:
            combined_image = image

        act = item["action"]
        obs = {
            "image": combined_image,  # (n_view, 3, 224, 224)
            "language_embedding": item["language_embedding"],
            "language_mask": item["language_mask"],
        }
        # Use low-dim state
        if "eef_states" in item["observation"].keys():
            eef_states = item["observation"]["eef_states"].astype(np.float32)
            obs["eef_states"] = eef_states

        return {"x0": act, "condition_cfg": obs}


class T5LanguageEncoder:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "google-t5/t5-base",
        max_length: int = 32,
        device: str = "cpu",
    ):
        from transformers import T5EncoderModel, T5Tokenizer

        self.device = device
        self.max_length = max_length
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = T5EncoderModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

    @torch.no_grad()
    def encode(self, sentences: List[str]):
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)
        return last_hidden_states, attention_mask

    def __call__(self, sentences: List[str]):
        return self.encode(sentences)
