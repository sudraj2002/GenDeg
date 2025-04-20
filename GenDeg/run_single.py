from __future__ import annotations

import math
import os
import random
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from torch import autocast
from torch.utils.data import DataLoader

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config
from edit_dataset import *

from NAFNet.basicsr.models.archs.NAFNet_arch import NAFNet

class NAFNet_Combine(NAFNet):
    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        # weight of SCM
        weight = 1
        x = x * weight + inp[:, 3:6, :, :]

        return x[:, :, :H, :W]

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "b ... -> (3 b) ...")
        cfg_sigma = einops.repeat(sigma, "b ... -> (3 b) ...")
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model."):]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def main():
    parser = ArgumentParser()
    # parser.add_argument("--input", required=True, type=str)  # Dataset directory
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--s", default=False, action='store_true', help='Whether to use structure correction')
    parser.add_argument("--auto_s", default=False, action='store_true',
                        help='Automatically use structure correction for haze, motion blur, raindrop.')
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/gendeg.ckpt", type=str)
    parser.add_argument("--img", default="/path/to/img", type=str, required=True)
    parser.add_argument("--prompt", default="prompt", type=str, required=True)
    parser.add_argument("--ckpt_s", default="checkpoints/nafnet_sd.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--deg_type", required=False, type=str, default="", choices=['haze', 'rain', 'snow', 'motion',
                                                                        'low-light', 'raindrop', ""])
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--mu", default=None)
    parser.add_argument("--sigma", default=None)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.mu is None:
        assert args.deg_type != "", \
            "Please provide --deg_type if you want to randomly sample mu and sigma. Its just the degradation you want to generate."


    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)

    # Null conditioning
    null_token = model.get_learned_conditioning([""])
    null_prompt_musigma = torch.zeros(args.batch_size, 2, 129).to(model.device)
    null_prompt_musigma[:, :, -1] = 1.0

    # Sturcture correction
    if args.s or args.auto_s:
        s = NAFNet_Combine(img_channel=6, width=64, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2])
        state_dict_s = torch.load(args.ckpt_s, map_location='cpu')
        s.load_state_dict(state_dict_s['state_dict'], strict=True)
        s.eval().cuda()

    # Single image dataset
    # Create dataset and dataloader
    dataset = EditDatasetSingle(prompt=args.prompt, name=args.img, split="test", deg_type=args.deg_type, res=args.resolution,
                                         mu=args.mu, sigma=args.sigma)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_idx, batch in enumerate(dataloader):
        name = batch['name']
        prompts = batch['edit']['c_crossattn']  # Prompts for the images
        stats = batch['edit']['c_stats'].to(model.device)
        edited_images = batch['edit']['c_concat'].to(model.device)  # Conditions
        mu = batch['mu']
        sigma = batch['sigma']

        with torch.no_grad(), autocast("cuda"), model.ema_scope():

            # COnditional guidance
            cond = {}
            stats_projected = model.stats_proj(stats)
            stats_reshaped = stats_projected.contiguous().transpose(1, 2)
            clip_text_emb = model.get_learned_conditioning(prompts)

            cond["c_crossattn"] = [model.cc_projection(torch.cat([clip_text_emb, stats_reshaped], dim=-1))]
            cond["c_concat"] = [model.encode_first_stage(edited_images).mode()]


            # Unconditional guidance
            null_prompt_mu_sigma_proj = model.stats_proj(null_prompt_musigma)
            null_prompt_mu_sigma_proj = null_prompt_mu_sigma_proj.contiguous().transpose(1, 2)

            uncond = {}
            uncond["c_crossattn"] = [model.cc_projection(torch.cat([null_token.repeat(args.batch_size, 1, 1),
                                                                    null_prompt_mu_sigma_proj.to(model.device)], dim=-1))]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = model_wrap.get_sigmas(args.steps)
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": args.cfg_text,
                "image_cfg_scale": args.cfg_image,
            }

            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = model.decode_first_stage(z).clamp_(-1, 1)
            if args.s:
                model_input = torch.cat((edited_images, x), dim=1)
                x = s(model_input)
            elif args.auto_s:
                if args.deg_type.lower() in ['haze', 'motion', 'raindrops']:
                    model_input = torch.cat((edited_images, x), dim=1)
                    x = s(model_input)

            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = (255.0 * rearrange(x, "b c h w -> b h w c")).type(torch.uint8).cpu().numpy()
            os.makedirs('result/', exist_ok=True)
            for i, image_array in enumerate(x):
                mu_i = mu[i]
                sigma_i = sigma[i]

                save_dir = os.path.join('result', f'mu{mu_i}_sigma{sigma_i}_{name[0]}')
                edited_image = Image.fromarray(image_array)
                edited_image.save(save_dir)


if __name__ == "__main__":
    main()
