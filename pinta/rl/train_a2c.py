#!/usr/bin/env python3

import argparse
import math
import os
import time

import gym
import numpy as np
import ptan
import torch
import torch.nn.functional as F
import torch.optim as optim
import pinta.rl.rl_continuous as model
import pinta.rl.rl_common as common
from torch.utils.tensorboard import SummaryWriter

HIDDEN_SIZE = 128
BATCH_SIZE = 64
PERCENTILE = 70


GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
ENTROPY_BETA = 1e-4

TEST_ITERS = 1e4


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            env.render()
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_logprob(mu_v, var_v, actions_v):
    p1 = -((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument(
        "-dnn", "--use_dnn", default=False, action="store_true", help="Use a machine learnt environment"
    )
    parser.add_argument("-cuda", "--cuda", default=False, action="store_true", help="Enable CUDA")

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    save_path = os.path.join("saves", "a2c-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env_args = {
        "white_noise": 0.03,
        "slow_moving_noise": 0.1,
        "inertia": 0.8,
        "target_twa": 0.8,
        "max_rudder": 1.0,
        "max_iter": 200,
        "model_path": "rnn_seq_27_hidden_128_batch_10000_lr_0.01_ep_40_amp_True.pt",
    }

    if args.use_dnn:
        env = gym.make("PintaEnv-v0", **env_args)
        test_env = gym.make("PintaEnv-v0", **env_args)
    else:
        env = gym.make("SimpleStochasticEnv-v0", **env_args)
        test_env = gym.make("SimpleStochasticEnv-v0", **env_args)

    net = model.ModelA2C(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    print(net)

    writer = SummaryWriter(comment="-a2c_" + args.name)
    agent = model.AgentA2C(net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if step_idx > 0 and step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(net, test_env, device=device)
                    print("Test done is %.2f sec, reward %.3f, steps %d" % (time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net.state_dict(), fname)
                        best_reward = rewards

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = common.unpack_batch_a2c(
                    batch, net, last_val_gamma=GAMMA ** REWARD_STEPS, device=device
                )
                batch.clear()

                optimizer.zero_grad()
                mu_v, var_v, value_v = net(states_v)

                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
                loss_policy_v = -log_prob_v.mean()
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2 * math.pi * var_v) + 1) / 2).mean()

                loss_v = loss_policy_v + entropy_loss_v + loss_value_v
                loss_v.backward()
                optimizer.step()

                summary = {
                    "step": step_idx,
                    "advantage": adv_v,
                    "values": value_v,
                    "batch_rewards": vals_ref_v,
                    "loss_entropy": entropy_loss_v,
                    "loss_policy": loss_policy_v,
                    "loss_value": loss_value_v,
                    "loss_total": loss_v,
                }

                for k in filter(lambda x: x != "step", summary.keys()):
                    tb_tracker.track(k, summary[k], summary["step"])

    env.close()
