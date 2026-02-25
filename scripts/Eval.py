# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--experiment_name", type=str, default=None, help="Experiment_name for logging.")


parser.add_argument("--targetObject", type=str, default=None, help="The object being tested.")
parser.add_argument("--envSeed", type=int, default=42, help="Seed used for the environment")

parser.add_argument(
    "--Full", action="store_true", default=False, help="Whether to include full trajectories in output."
)

parser.add_argument("--ctrlFreq", type=int, default=None, help="Control frequency for the environment.")
parser.add_argument("--jerkLimit", type=float, default=None, help="Jerk limit for the environment.")
parser.add_argument("--histLen", type=int, default=None, help="History length for the environment.")

parser.add_argument("--numEpochs", type=int, default=None, help="Number of epochs for training.")

parser.add_argument("--DOF", type=int, default=None, help="Degrees of freedom for the environment.")

parser.add_argument("--controlMode", type=str, default=None, help="Control mode for the environment.")



# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import torch
import numpy as np
import time
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import Throw.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper


def save_images_to_gif(images, filename):
    images[0].save(filename, save_all=True, append_images=images[1:], loop=0, duration=17)


def append_fifo(fifo, data):
    fifo = torch.cat((fifo[1:], torch.unsqueeze(data, dim=0)), dim=0)
    return fifo


def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    experiment_name = args_cli.experiment_name if args_cli.experiment_name is not None else ""
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        name = agent_cfg["params"]["config"]["name"]
        seed = args_cli.seed
        run_dir = (
            f"{experiment_name}_seed{seed}"  # agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        )
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    if args_cli.ctrlFreq is not None:
        print(f"[INFO] Setting control frequency to {args_cli.ctrlFreq} Hz.")
        env_cfg.ctrl_Freq = args_cli.ctrlFreq
        env_cfg.decimation = int(env_cfg.sim_Freq / args_cli.ctrlFreq)
        env_cfg.maxEp_steps = int(env_cfg.TrajTime*args_cli.ctrlFreq) # 48*3=144
        env_cfg.max_Steps = int(env_cfg.episode_length_s * args_cli.ctrlFreq)  # freq(120)/decimation(2)
        env_cfg.contact_sensor.update_period=1/args_cli.ctrlFreq


    if args_cli.jerkLimit is not None:
        print(f"[INFO] Setting jerk limit to {args_cli.jerkLimit}.")
        env_cfg.jerkLimit = args_cli.jerkLimit

    if args_cli.histLen is not None:
        print(f"[INFO] Setting history length to {args_cli.histLen}.")
        env_cfg.histLen = args_cli.histLen
        env_cfg.observation_space =  7 + (env_cfg.DOF*4)*args_cli.histLen +3 #+1#+ 9 
        env_cfg.state_space =   38 + (env_cfg.DOF*4)*args_cli.histLen + 3 + 1 +3*env_cfg.RandCOM #+1 # 31 + 1 + 2 + 6 + 3

    if args_cli.numEpochs is not None:
        print(f"[INFO] Setting max epochs to {args_cli.numEpochs}.")
        agent_cfg["params"]["config"]["max_epochs"] = args_cli.numEpochs
    
    if args_cli.DOF is not None:
        print(f"[INFO] Setting DOF to {args_cli.DOF}.")
        env_cfg.DOF = args_cli.DOF
        env_cfg.action_space = env_cfg.DOF  # spaces.Box(-2, 2, shape=(6,))
        env_cfg.observation_space =  7 + (env_cfg.DOF*4)*env_cfg.histLen +3 #+1#+ 9 
        env_cfg.state_space =   38 + (env_cfg.DOF*4)*env_cfg.histLen + 3 + 1 +3*env_cfg.RandCOM #+1 # 31 + 1 + 2 + 6 + 3
    
    if args_cli.controlMode is not None:
        print(f"[INFO] Setting control mode to {args_cli.controlMode}.")
        env_cfg.actionMode = args_cli.controlMode
        
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    env.seed(args_cli.envSeed)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        state = obs["states"]
        obs = obs["obs"] 


    target = args_cli.targetObject 

    # reset environment
    success_len = 5
    n_envs = env.unwrapped.num_envs #512
    nsteps = 1800
    #OBS = torch.zeros((nsteps+1, n_envs, state.shape[1]), device="cuda")
    #ACTION = torch.zeros((nsteps+1, n_envs, 6), device="cuda")
    #SUCCESS = torch.zeros((success_len+1, n_envs), device="cuda")
    #STATES = torch.zeros((nsteps+1, n_envs, 5), device="cuda")

    # required: enables the flag for batched observations
    print("[INFO] Getting batch size.", obs.shape)
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        print("[INFO] Initializing RNN states.")
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    OBS = []
    ACTION = []
    SUCCESS = []
    STATES = []
    JOINTACT = []
    Times = []
    for ind in range(nsteps):
        with torch.inference_mode():
            # convert obs to agent format
            if isinstance(obs, dict):
                state = obs["states"]
                obs = obs["obs"]
                #print("OBS joint_q", obs[0, :6])
                #exit()

            OBS.append(state)
            # agent stepping
            actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            #actions = torch.zeros_like(actions)  # set actions to zero
            ACTION.append(actions)
            # env stepping
            obs, _, dones, extras = env.step(actions)

            if dones.all():
                cmdJerk = extras["episode"]["cmdJerk"]
                JOINTACT.append(cmdJerk)

                success = extras["episode"]["is_success"].to(int)
                SUCCESS.append(success)

                block_state = extras["episode"]["block_state"]
                ee_state = extras["episode"]["ee_state"]
                joint_state = extras["episode"]["joint_state"]
                target_pos = extras["episode"]["target_pos"]
                object_model = extras["episode"]["ObjectModel"]


                states = torch.cat(
                    [block_state, ee_state, joint_state, target_pos, object_model], dim=-1
                )
                STATES.append(states)

                break
            success = extras["episode"]["is_success"].to(int)
            SUCCESS.append(success)

            cmdJerk = extras["episode"]["cmdJerk"]
            JOINTACT.append(cmdJerk)


            block_state = extras["episode"]["block_state"]
            ee_state = extras["episode"]["ee_state"]
            joint_state = extras["episode"]["joint_state"]
            target_pos = extras["episode"]["target_pos"]
            object_model = extras["episode"]["ObjectModel"]


            states = torch.cat(
                [block_state, ee_state, joint_state, target_pos, object_model], dim=-1
            )
            STATES.append(states)


            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0

    OBS = torch.stack(OBS, dim=0)
    ACTION = torch.stack(ACTION, dim=0)
    SUCCESS = torch.stack(SUCCESS, dim=0)
    JOINTACT = torch.stack(JOINTACT, dim=0)
    STATES = torch.stack(STATES, dim=0)
    is_success = SUCCESS[-success_len - 1 : -1, :].sum(dim=0) == success_len
    logTraj = is_success
    print(f"seed {seed} -- success rate: {sum(logTraj).item()/n_envs*100:.2f}%")
    Summary = torch.cat((is_success.reshape(-1, 1), STATES[0]), dim=-1)

    

    envSeed = args_cli.envSeed
    os.makedirs(f"results/rl_games/{name}/{experiment_name}/{target}/seed{envSeed}", exist_ok=True)
    summary_path = f"results/rl_games/{name}/{experiment_name}/{target}/seed{envSeed}/summary.txt"
    np.savetxt(summary_path, Summary.cpu().numpy(), delimiter=",")
    
    if args_cli.Full:
        for i in range(n_envs):
            traj_ind = i
            os.makedirs(f"results/rl_games/{name}/{experiment_name}/{target}/seed{envSeed}/{traj_ind:03d}", exist_ok=True)

            actions_traj_path = (
                f"results/rl_games/{name}/{experiment_name}/{target}/seed{envSeed}/{traj_ind:03d}/action_traj.txt"
            )
            actions_traj = ACTION[:, i].cpu().numpy()
            np.savetxt(actions_traj_path, actions_traj, delimiter=",")

            jointacc_traj_path = (
                f"results/rl_games/{name}/{experiment_name}/{target}/seed{envSeed}/{traj_ind:03d}/jointact_traj.txt"
            )
            jointact_traj = JOINTACT[:, i].cpu().numpy()
            np.savetxt(jointacc_traj_path, jointact_traj, delimiter=",")

            obs_traj_path = f"results/rl_games/{name}/{experiment_name}/{target}/seed{envSeed}/{traj_ind:03d}/obs_traj.txt"
            obs_traj = OBS[:, i].cpu().numpy()
            np.savetxt(obs_traj_path, obs_traj, delimiter=",")

            success_traj_path = (
                f"results/rl_games/{name}/{experiment_name}/{target}/seed{envSeed}/{traj_ind:03d}/success_traj.txt"
            )
            success_traj = SUCCESS[-success_len:, i].cpu().numpy()
            np.savetxt(success_traj_path, success_traj, delimiter=",")

            states_traj_path = (
                f"results/rl_games/{name}/{experiment_name}/{target}/seed{envSeed}/{traj_ind:03d}/states_traj.txt"
            )
            states_traj = STATES[:, i].cpu().numpy()
            np.savetxt(states_traj_path, states_traj, delimiter=",")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
