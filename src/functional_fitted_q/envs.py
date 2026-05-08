from os import path
from typing import Optional

import gym
import numpy as np
import pandas as pd
import torch
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from tqdm import tqdm

from .common import DEFAULT_X, DEFAULT_Y, calculate_reward, calculate_reward_c, generate_action, random_policy


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class PendulumEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.01
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        # Advance the pendulum by one low-level torque input.
        th, thdot = self.state
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l**2) * u) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -costs, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        low = -high
        self.state = np.random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                f'You can specify the render_mode at initialization, e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled("pygame is not installed, run `pip install gym[classic_control]`")

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
            else:
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2
        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]

        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            transformed_coords.append((c[0] + offset, c[1] + offset))
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))

        rod_end = pygame.math.Vector2((rod_length, 0)).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77))

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname) if path.exists(fname) else None
        if self.last_u is not None and img is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            scale_img = pygame.transform.flip(scale_img, bool(self.last_u > 0), True)
            self.surf.blit(
                scale_img,
                (offset - scale_img.get_rect().centerx, offset - scale_img.get_rect().centery),
            )

        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class PendulumEnv_c(PendulumEnv):
    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        super().__init__(render_mode=render_mode, g=g)
        self.dt = 1


def _rollout_episode(env, cycle_num, x_length):
    data = []
    next_obs = None
    for cycle in range(cycle_num):
        obs = env.reset()[0] if cycle == 0 else next_obs
        initial_state = obs
        initial_action = random_policy(x_length=x_length)

        for step in range(x_length):
            action = np.array([initial_action[step]]) if step < len(initial_action) else np.array([0.0])
            next_obs, _, _, _, _ = env.step(action)

        data.append(
            [
                initial_state[0],
                initial_state[1],
                initial_state[2],
                initial_action,
                np.arctan2(initial_state[1], initial_state[0]),
                initial_state,
                next_obs,
            ]
        )
    return data


def Pendulum_data_generator(episode_num=40, cycle_num=10, x_length=100):
    # Each row stores one full functional action trajectory and the resulting next state.
    env = PendulumEnv()
    data = []
    for episode in range(episode_num):
        episode_rows = _rollout_episode(env, cycle_num, x_length)
        for row in episode_rows:
            data.append([episode, *row])

    df = pd.DataFrame(
        data,
        columns=["Subject", "cos(theta)", "sin(theta)", "thetadot", "FunctionalData", "theta", "State", "Next_State"],
    )
    df["Time"] = df.groupby("Subject").cumcount() + 1
    df["Status"] = df["Time"].apply(lambda x: x == cycle_num)
    df["Continuous"] = df["FunctionalData"].apply(lambda x: np.sign(np.mean(x)) * np.sqrt(np.mean(np.square(x))))
    df["Reward"] = df.apply(calculate_reward, axis=1)
    return df


def Pendulum_data_generator_c(episode_num=40, cycle_num=10, x_length=100):
    # This baseline version compresses the functional action into a single scalar summary.
    df = Pendulum_data_generator(episode_num=episode_num, cycle_num=cycle_num, x_length=x_length).copy()
    df["Reward"] = df.apply(calculate_reward_c, axis=1)
    return df


def Pendulum_data_generator_for_monte_c(episode_num=40, cycle_num=10, model=None, max_iterations=100, device=None):
    env = PendulumEnv_c()
    data = []
    next_obs = None

    for episode in range(episode_num):
        for cycle in range(cycle_num):
            obs = env.reset()[0] if cycle == 0 else next_obs
            action = model.get_policy(
                torch.tensor([obs], dtype=torch.float32, device=device),
                max_iterations=max_iterations,
            ).to("cpu").numpy().squeeze(0)
            next_obs, _, _, _, _ = env.step(action)
            data.append([episode, obs[0], obs[1], obs[2], action[0], np.arctan2(obs[1], obs[0])])

    df = pd.DataFrame(data, columns=["Subject", "cos(theta)", "sin(theta)", "thetadot", "Continuous", "theta"])
    df["Time"] = df.groupby("Subject").cumcount() + 1
    df["Reward"] = df.apply(calculate_reward_c, axis=1)
    return df


def Pendulum_data_generator_for_monte(episode_num=40, cycle_num=10, x_length=100, model=None, device=None):
    # Roll out the learned functional policy to estimate its discounted return by simulation.
    env = PendulumEnv()
    data = []
    next_obs = None

    for episode in range(episode_num):
        for cycle in range(cycle_num):
            obs = env.reset()[0] if cycle == 0 else next_obs
            initial_state = obs
            initial_action = model.get_policy(torch.tensor([initial_state], dtype=torch.float32, device=device)).to("cpu").numpy().squeeze()

            for step in range(x_length):
                action = np.array([initial_action[step]]) if step < len(initial_action) else np.array([0.0])
                next_obs, _, _, _, _ = env.step(action)

            data.append([episode, initial_state[0], initial_state[1], initial_state[2], initial_action, np.arctan2(initial_state[1], initial_state[0])])

    df = pd.DataFrame(data, columns=["Subject", "cos(theta)", "sin(theta)", "thetadot", "FunctionalData", "theta"])
    df["Time"] = df.groupby("Subject").cumcount() + 1
    df["Reward"] = df.apply(calculate_reward, axis=1)
    return df


def Pendulum_data_generator_for_policy(episode_num=40, cycle_num=10, x_length=100, policy=None):
    # Roll out a fixed stochastic policy directly, without fitting a model first.
    if policy is None:
        policy = generate_action

    env = PendulumEnv()
    data = []
    next_obs = None

    for episode in range(episode_num):
        for cycle in range(cycle_num):
            obs = env.reset()[0] if cycle == 0 else next_obs
            initial_state = obs
            initial_action = policy(initial_state)

            for step in range(x_length):
                action = np.array([initial_action[step]]) if step < len(initial_action) else np.array([0.0])
                next_obs, _, _, _, _ = env.step(action)

            data.append([episode, initial_state[0], initial_state[1], initial_state[2], initial_action, np.arctan2(initial_state[1], initial_state[0])])

    df = pd.DataFrame(data, columns=["Subject", "cos(theta)", "sin(theta)", "thetadot", "FunctionalData", "theta"])
    df["Time"] = df.groupby("Subject").cumcount() + 1
    df["Reward"] = df.apply(calculate_reward, axis=1)
    return df


def monte_carlo_value_score(sim_num=10, cycle_num=10, model=None, discount=0.99, device=None):
    rewards = []
    with tqdm(total=sim_num, desc="Threads") as pbar:
        for _ in range(sim_num):
            data = Pendulum_data_generator_for_monte(1, cycle_num=cycle_num, model=model, device=device)
            discount_factors = np.array([discount**i for i in range(len(data))])
            rewards.append([np.sum(discount_factors * data["Reward"])])
            pbar.update(1)
    return np.mean(rewards)


def monte_carlo_policy_value_score(sim_num=10, cycle_num=10, x_length=100, policy=None, discount=0.99):
    rewards = []
    with tqdm(total=sim_num, desc="Policy MC") as pbar:
        for _ in range(sim_num):
            data = Pendulum_data_generator_for_policy(
                1,
                cycle_num=cycle_num,
                x_length=x_length,
                policy=policy,
            )
            discount_factors = np.array([discount**i for i in range(len(data))])
            rewards.append([np.sum(discount_factors * data["Reward"])])
            pbar.update(1)
    return np.mean(rewards)


def monte_carlo_value_score_c(sim_num=10, cycle_num=10, model=None, max_iterations=2000, discount=0.99, device=None):
    rewards = []
    with tqdm(total=sim_num, desc="Threads") as pbar:
        for _ in range(sim_num):
            data = Pendulum_data_generator_for_monte_c(
                1,
                cycle_num=cycle_num,
                model=model,
                max_iterations=max_iterations,
                device=device,
            )
            discount_factors = np.array([discount**i for i in range(len(data))])
            rewards.append([np.sum(discount_factors * data["Reward"])])
            pbar.update(1)
    return np.mean(rewards)


__all__ = [
    "DEFAULT_X",
    "DEFAULT_Y",
    "PendulumEnv",
    "PendulumEnv_c",
    "Pendulum_data_generator",
    "Pendulum_data_generator_c",
    "Pendulum_data_generator_for_monte",
    "Pendulum_data_generator_for_monte_c",
    "Pendulum_data_generator_for_policy",
    "angle_normalize",
    "monte_carlo_policy_value_score",
    "monte_carlo_value_score",
    "monte_carlo_value_score_c",
]
