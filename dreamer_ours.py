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

import sys
sys.path.insert(0,'/home/gary/wm_robot')
from datasets.pusht_dset import PushTDataset, load_pusht_slice_train_val
from torch.utils.data import DataLoader

from tqdm import tqdm

to_np = lambda x: x.detach().cpu().numpy()

import torchvision.transforms as transforms

def resize_image(image_tensor): # resize the image to 64x64
    resize_transform = transforms.Resize((64, 64))
    resized_image = resize_transform(image_tensor)
    return resized_image

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = 1
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(iter(self._dataset)))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(iter(self._dataset)))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
        return metrics
    
    def train_one_epoch(self):
        # loop over the dataset
        all_metrics = []
        for batch in tqdm(self._dataset):
            cur_metrics = self._train(batch)
            all_metrics.append(cur_metrics)
        return all_metrics


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def main(config):
    tools.set_seed_everywhere(config.seed)
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir = logdir / "saved_models" / config.task / str(config.run_idx)
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    # step in logger is environmental step
    print("Action Space", config.action_dim )
    config.num_actions = 2

    # here we should load the dataset
    datasets, traj_dset = load_pusht_slice_train_val(data_path = '/data/jeff/workspace/pusht_dataset', with_velocity=False,n_rollout=5, transform=resize_image, frameskip=1, num_hist=config.batch_length)
    train_dataset = datasets['train']
    eval_dataset = datasets['valid']
    train_dataset = DataLoader(train_dataset,batch_size=config.batch_size, shuffle=True)
    eval_dataset = iter(DataLoader(eval_dataset,batch_size=config.batch_size, shuffle=True))
    agent = Dreamer(
        (64, 64, 3), # obs_space
        (config.action_dim ,), # act_space
        config,
        None,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists() and config.continue_training: # load the latest model if exists
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    accumulated_metric = []
    for epoch in tqdm(range(config.epochs)):
        if config.video_pred_log and epoch % config.save_video_every == 0:
            print("eval_openl")
            video_pred = agent._wm.video_pred(next(eval_dataset)) #!!! this is changed!
            single_video_pred = video_pred[0].cpu()
            single_video_pred = (single_video_pred * 255).clamp(0, 255).byte()
            tvio.write_video(logdir / f"eval_openl{epoch}.mp4", single_video_pred, fps=30, video_codec='mpeg4')
            
        all_metrics = agent.train_one_epoch()
        accumulated_metric.extend(all_metrics)
        if epoch % config.save_accumulated_metric_every == 0:
            data_to_save = accumulated_metric
            with open(logdir / 'metrics.pkl', 'wb') as f:
                pickle.dump(data_to_save, f)

        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / f"{epoch}.pt")
        torch.save(items_to_save, logdir / "latest.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
