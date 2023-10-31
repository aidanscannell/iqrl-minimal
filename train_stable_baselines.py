#!/usr/bin/env python3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import gymnasium as gym
import omegaconf
import wandb
from hydra.utils import get_original_cwd
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
import hydra


# def make_env(env_id):
#     def thunk():
#         env = gym.make(env_id)
#         env = Monitor(env)  # record stats such as returns
#         return env

#     return thunk


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


@hydra.main(
    version_base="1.3", config_path="./configs", config_name="train_stable_baselines"
)
# def train(cfg: TrainConfig):
def train(cfg):
    # config = {
    #     "policy_type": "MlpPolicy",
    #     "total_timesteps": 25000,
    #     "env_name": "CartPole-v1",
    # }
    # Initialise WandB
    if cfg.use_wandb:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project_name,
            # entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            tags=cfg.wandb_tags,
            # sync_tensorboard=True,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
            name=cfg.run_name,
            monitor_gym=cfg.monitor_gym,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )
    # run = wandb.init(
    #     project="sb3",
    #     config=cfg,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     monitor_gym=True,  # auto-upload the videos of agents playing the game
    #     save_code=True,  # optional
    # )

    env = DummyVecEnv(
        [
            make_env(
                cfg.env_id,
                seed=cfg.seed,
                idx=0,
                capture_video=True,
                run_name=cfg.run_name,
            )
        ]
    )
    # env = DummyVecEnv([make_env(cfg.env_id)])
    # env = VecVideoRecorder(
    #     env,
    #     f"videos/{run.id}",
    #     record_video_trigger=lambda x: x % 2000 == 0,
    #     video_length=200,
    # )
    model = hydra.utils.instantiate(
        cfg.agent,
        env=env,
        tensorboard_log=f"runs/{run.id}",
    )
    # model = TD3(
    #     cfg.policy_type,
    #     env,
    #     verbose=1,
    #     tensorboard_log=f"runs/{run.id}",
    # )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_log_dir,
        log_path=eval_log_dir,
        eval_freq=max(500 // n_training_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=WandbCallback(
            # gradient_save_freq=100,
            # model_save_path=f"models/{run.id}",
            # verbose=2,
        ),
    )
    run.finish()


if __name__ == "__main__":
    train()  # pyright: ignore
