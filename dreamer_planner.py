import argparse
import functools
import os
import pathlib
import sys
import pickle

import torchvision.io as tvio

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()

import torchvision.transforms as transforms

import einops


class DreamerPlanner(nn.Module):
    def __init__(self, wm, config):
        super(DreamerPlanner, self).__init__()
        self._wm = wm
        self._config = config
        self.encoder = 1
        self.decoder = 1
 
    @torch.no_grad()
    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        assert len(z_obs['visual'].shape) == 4
        assert z_obs['visual'].shape[2] == 1
        z_obs = z_obs['visual'].squeeze(2)
        decoded = self._wm.heads["decoder"](z_obs)
        decoded_img = decoded["image"].mode() # b t h w c
        return [{'visual':einops.rearrange(decoded_img, "b t h w c -> b t c h w")}]

    @torch.no_grad()
    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        obs = self.preprocess_obs(obs)
        z = self._wm.encoder(obs)
        if len(z.shape) == 3:
            z = z.unsqueeze(2)
        return {'visual': z}
        
    
    @torch.no_grad()
    def rollout(self, obs_0, act, step_size=1):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        is_first = torch.zeros(act.shape[:2], device=self._config.device)
        b, n = obs_0["visual"].shape[:2]
        assert n >= 1
        t = act.shape[1] - n
        embed_0 = self.encode_obs(obs_0)['visual'] # (b, n, 1, emb_dim)
        embed_0 = embed_0.squeeze(2) # (b, n, emb_dim)
        prev_act = act[:, :n]
        future_act = act[:, n:]
        states, _ = self._wm.dynamics.observe(
            embed_0, prev_act, is_first[:, :n]
        )
        recon = self._wm.heads["decoder"](self._wm.dynamics.get_feat(states))["image"].mode()
        recon = torch.cat([recon[:, 0:1], recon], dim=1) # repeat the first frame of recon
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self._wm.dynamics.imagine_with_action(future_act, init)
        decoded_predict = self._wm.heads["decoder"](self._wm.dynamics.get_feat(prior))["image"].mode()
        decoded_predict = torch.cat([recon, decoded_predict], dim=1)
        decoded_predict = einops.rearrange(decoded_predict, "b t h w c -> b t c h w")
        encoded_decoded_predict = self.encode_obs({"visual": decoded_predict})['visual']
        encoded_decoded_predict = {'visual': encoded_decoded_predict}
        return encoded_decoded_predict, decoded_predict
        
    
    def preprocess_obs(self, obs):
        obs = obs['visual']
        if obs.shape[-1] != 3:
            if len(obs.shape) == 4:
                obs = einops.rearrange(obs, "b c h w -> b h w c")
            else:
                obs = einops.rearrange(obs, "b t c h w -> b t h w c")
        obs = {
            "image": obs,
        }
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        return obs