# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    matrix_from_quat,
    subtract_frame_transforms)

from isaaclab.markers import VisualizationMarkers

from .throw_env_cfg import ThrowEnvCfg


class ThrowEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: ThrowEnvCfg

    def __init__(self, cfg: ThrowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._device = self.sim.device
        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
    
        self.robot_default_jointPos = self._robot.data.default_joint_pos[:, :].clone()

        self.tray_link_idx = self._robot.find_bodies("tray")[0][0]
        self.robot_root_pos_w = self._robot.data.root_link_state_w[:, :3]
        self.robot_root_quat_w = self._robot.data.root_link_state_w[:, 3:7]
        self.default_quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat((self.num_envs, 1))

        assets_sizes = torch.tensor(self.cfg.assets_sizes, device=self.device, dtype=torch.float32)
        self.assets_sizes = assets_sizes[:self.num_envs]

        self.real_assets_sizes = self.assets_sizes.clone()

        self.init_state_buffers()

        self.scale = 0.0
        


        self.target_pos = torch.tensor([2.0,0.0,0]).repeat(self.num_envs, 1).to(self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._stand = RigidObject(self.cfg.stand)
        self._bin = RigidObject(self.cfg.bin)
        self._custom_ground = RigidObject(self.cfg.custom_ground)

        self._block = RigidObject(self.cfg.block)
        self.scene.rigid_objects["block"] = self._block

        self._contactSensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contactSensor

        self.frame_markers = VisualizationMarkers(self.cfg.frame_object_cfg)

        self.goal_marker = VisualizationMarkers(self.cfg.goal_object_cfg)


        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.5, 0.5, 0.5))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process Jerk actions from  to joint position"""

        dt = self.cfg.sim.dt*self.cfg.decimation

        if self.cfg.planar:
            self.actions[:,1:4] = actions[:,:3].clone()
        else:
            self.actions = actions.clone()
        #post_throw_hat = (torch.sigmoid(actions[:,3]) > 0.5).clone().reshape(-1, 1).repeat(1, 6)

        self.actions = self.last_action + self.cfg.AccRate*self.actions

        self.actions = torch.clamp(self.actions, -1.0, 1.0)
        if self.cfg.planar:
            self.actions_fifo = torch.cat([self.actions_fifo[:,1:,:], self.actions[:,1:4].clone().reshape(-1, 1, 3)], dim=1)
        else:
            self.actions_fifo = torch.cat([self.actions_fifo[:,1:,:], self.actions.clone().reshape(-1, 1, 6)], dim=1)
        #self.cmd_acc = self.actions* self.cfg.action_scale

        steps_since_postThrow = self.steps_since_postThrow.reshape(-1, 1).repeat(1, 6)
        post_throw = (steps_since_postThrow > 0)
        #relStep = (self.episode_length_buf % self.cfg.maxEp_steps).float() / self.cfg.maxEp_steps
        #post_throw = (relStep.reshape(-1, 1).repeat(1, 6)) > 

        #self.post_throw_error = torch.linalg.norm(post_throw_hat.float() - post_throw.float(), dim=-1)

        # stop the robot by aggressively decelerating with additional damping
        damping_factor = 5.0  # Increase this for stronger damping
        acc = -self.joint_qd * (1 / dt) * 0.5 # Add velocity-based damping (1/dt=20)
        max_delta_acc = self.cfg.action_scale * self.cfg.AccRate  # Increased from 5 for faster deceleration
        delta_acc = self.cmd_acc - acc
        small_acc = (torch.abs(delta_acc) < max_delta_acc).float()
        clipped_acc = self.cmd_acc - torch.clip(delta_acc, min=-max_delta_acc, max=max_delta_acc)
        acc = acc * small_acc + (1-small_acc) * clipped_acc
        self.cmd_acc = self.actions * self.cfg.action_scale * (1-post_throw.float()) + acc * post_throw.float()


        vel_est = self.joint_qd + self.cmd_acc * dt
        max_vel = 3.0
        vel_above = (vel_est > max_vel).float()
        vel_below = (vel_est < -max_vel).float()
        high_vel_est = vel_above + vel_below
        acc_to_vel_limit = (max_vel - self.joint_qd) / dt
        acc_to_neg_limit = (-max_vel - self.joint_qd) / dt
        clipped_acc = vel_above * acc_to_vel_limit + vel_below * acc_to_neg_limit
        self.cmd_acc = self.cmd_acc.clone()* (1-high_vel_est) + high_vel_est*clipped_acc

        self.cmd_acc = torch.clamp(self.cmd_acc, -self.cfg.action_scale, self.cfg.action_scale)
        
        self.last_action = self.actions.clone()

            
    def _apply_action(self):
        """Apply the joint position targets to the robot"""
        dt = self.cfg.sim.dt 

        
        self.joint_q = self.joint_q + self.joint_qd * (dt) + 0.5 * self.cmd_acc * (dt) ** 2 #
        self.joint_qd = self.joint_qd + self.cmd_acc*(dt)

        commanded_vel = torch.clamp(self.joint_qd, -3, 3)
        self._robot.set_joint_velocity_target(commanded_vel)
    


    # post-physics step calls
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        truncated = self.episode_length_buf >= (self.max_episode_length)
        if self.cfg.evaluating:
            terminated = truncated.clone()
        else:
            dropped = self.block_pos_w[:, 2] < 0.0
            eePos_x = self.eePos_w[:, 0] - self.robot_root_pos_w[:, 0]
            far_ee_x = eePos_x < -0.5
            terminated = dropped | far_ee_x

        self.num_terminated = torch.sum(terminated).item()
        return terminated, truncated

    ################################################## RESET #######################################################
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        
        self.sample_env_state(env_ids)  # Target Positions

        self.last_action[env_ids] = torch.zeros_like(self.last_action[env_ids])
        self.actions_fifo[env_ids] = torch.zeros_like(self.actions_fifo[env_ids])
        self.steps_since_postThrow[env_ids] = 0
        
        self.joint_q[env_ids] = self._robot.data.joint_pos[env_ids].clone()
        self.joint_qd[env_ids] = self._robot.data.joint_vel[env_ids].clone()

        if self.cfg.planar:
            self.joint_pos_fifo[env_ids] = self.joint_q[env_ids,1:4].clone().reshape(-1, 1, 3).repeat(1, self.cfg.histLen, 1)
        else:
            self.joint_pos_fifo[env_ids] = self.joint_q[env_ids,:].clone().reshape(-1, 1, 6).repeat(1, self.cfg.histLen, 1)
        self.joint_vel_fifo[env_ids] = torch.zeros_like(self.joint_pos_fifo[env_ids])


        self.block_mass = self._block.root_physx_view.get_masses().clone().to(device=self.device)
        block_props= self._block.root_physx_view.get_material_properties().clone().to(device=self.device)
        self.block_staticFriction  = block_props[:, 0, 0].reshape(-1, 1)
        self.block_dynamicFriction = block_props[:, 0, 1].reshape(-1, 1)
        self.block_restitution = block_props[:, 0, 2].reshape(-1, 1)
        self.block_com = self._block.root_physx_view.get_coms().clone()[:,:3].to(device=self.device)


        self.joint_pos = self.joint_q.clone()
        self.joint_vel = self.joint_qd.clone()

        block_state = self._block.data.body_link_state_w[env_ids, 0].clone()
        self.block_pos_w[env_ids] = block_state[:, :3]
        self.block_quat_w[env_ids] = block_state[:, 3:7]
        self.block_vel[env_ids] = block_state[:, 7:]

        ee_state = self._robot.data.body_link_state_w[env_ids, self.tray_link_idx].clone()
        self.eePos_w[env_ids] = ee_state[:, :3]
        self.eeQuat_w[env_ids] = ee_state[:, 3:7]
        self.eeVel[env_ids] = ee_state[:, 7:]

        # Randomize object parameters for robustness analysis
        #self.block_staticFriction[env_ids] = torch.ones_like(self.block_staticFriction[env_ids]) * 1.0 # base 0.5
        #self.block_dynamicFriction[env_ids] = torch.ones_like(self.block_dynamicFriction[env_ids]) * 0.1 # base 0.
        #self.block_restitution[env_ids] = torch.ones_like(self.block_restitution[env_ids]) * 0.2   # base 0.25
        
        #self.block_mass[env_ids] = torch.ones_like(self.block_mass[env_ids]) * 0.1 # base 0..5
        #self.assets_sizes[env_ids,0] = torch.ones_like(self.assets_sizes[env_ids,0])*0.05
        #self.assets_sizes[env_ids,1] = torch.ones_like(self.assets_sizes[env_ids,1])*0.1
        #self.assets_sizes[env_ids,2] = torch.ones_like(self.assets_sizes[env_ids,2])*0.1      # base 0.1

        self.update_curriculum()

    def compute_observation(self, target="policy") -> torch.Tensor:
        
        relStep = (self.episode_length_buf % self.cfg.maxEp_steps).float() / self.cfg.maxEp_steps
        relStep = relStep.reshape(-1, 1)


        ee_pos = self.eePos_w - self.robot_root_pos_w
        block_pos = self.block_pos_w - self.robot_root_pos_w
        
        #relative block pose to the tray
        (block_pos_rel, block_quat_rel) = subtract_frame_transforms(ee_pos, self.eeQuat_w, block_pos, self.block_quat_w)
        block_pos =  block_pos_rel
        block_quat = block_quat_rel

        eeQuat_w =  self.eeQuat_w
        
        ee_Vel = self.eeVel
        block_vel = self._block.data.body_link_vel_w[:, 0, :].clone()

        ee_rotMat = matrix_from_quat(eeQuat_w)
        ee_rotMat = ee_rotMat[:, :, :2].reshape(-1, 6)
        ee_ori = ee_rotMat


        block_rotMat = matrix_from_quat(block_quat)
        block_rotMat = block_rotMat[:, :, :2].reshape(-1, 6)
        block_ori = block_rotMat


        block_state = torch.cat([block_pos, block_ori], dim=-1) # type: ignore
        if self.cfg.planar:
            actions_fifo = self.actions_fifo.reshape(-1, 3*self.cfg.histLen) if self.cfg.histLen > 1 else self.actions[:,1:4].clone()
            joint_pos_fifo = self.joint_pos_fifo.reshape(-1, 3*self.cfg.histLen) if self.cfg.histLen > 1 else self.joint_q[:,1:4].clone()
            joint_vel_fifo = self.joint_vel_fifo.reshape(-1, 3*self.cfg.histLen) if self.cfg.histLen > 1 else self.joint_qd[:,1:4].clone()
        else:
            actions_fifo = self.actions_fifo.reshape(-1, 6*self.cfg.histLen) if self.cfg.histLen > 1 else self.actions.clone()
            joint_pos_fifo = self.joint_pos_fifo.reshape(-1, 6*self.cfg.histLen) if self.cfg.histLen > 1 else self.joint_q.clone()
            joint_vel_fifo = self.joint_vel_fifo.reshape(-1, 6*self.cfg.histLen) if self.cfg.histLen > 1 else self.joint_qd.clone()

        steps_since_postThrow = self.steps_since_postThrow.reshape(-1, 1)
        post_throw = (steps_since_postThrow > 2).float()

        if target == "policy":
            obs = torch.cat(
                (
                    joint_pos_fifo,          # 6 +
                    joint_vel_fifo,          # 6 + --> 12
                    self.assets_sizes,  # 3 + --> 18
                    actions_fifo,
                    self.block_mass,
                    self.block_staticFriction, # 1 + --> 26
                    self.block_dynamicFriction, # 1 + --> 27
                    self.block_restitution, # 1 + --> 82
                    relStep,
                    self.target_pos, # 2D or 3D (currently a redundant y axis)
                ),
                dim=-1,
            )

            
        if target == "critic":

            
            obs = torch.cat(
                (
                    joint_pos_fifo,      # 6 +
                    joint_vel_fifo,      # 6 + --> 12
                    ee_pos,        # 3 + --> 15
                    ee_ori,         # 6 + --> 21
                    ee_Vel,         # 6 + --> 27
                    block_state,    # 9 + --> 36
                    block_vel,      # 6 + --> 42
                    self.assets_sizes,  # 3 + --> 48
                    actions_fifo,
                    self.block_mass,
                    self.block_staticFriction, # 1 + --> 80
                    self.block_dynamicFriction, # 1 + --> 81
                    self.block_restitution, # 1 + --> 82
                    relStep,
                    self.target_pos,
                    post_throw,
                ),
                dim=-1,
            )

            if self.cfg.RandCOM:
                obs = torch.cat((obs, self.block_com), dim=-1)

        return obs.clone()
    
    def _get_observations(self) -> dict:
        obs = self.compute_observation("policy")
        state = self.compute_observation("critic")        
        self.frame_markers.visualize(self.block_pos_w, self.block_quat_w)
        target_pos = self.target_pos + self._robot.data.root_link_state_w[:, :3]
        self.goal_marker.visualize(target_pos, self.default_quat)
        observations = {"policy": obs, "critic": state}

        return observations

    # auxiliary methods
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        eeState_w = self._robot.data.body_link_state_w[env_ids, self.tray_link_idx].clone()
        self.eePos_w[env_ids] = eeState_w[:, :3]
        self.eeQuat_w[env_ids] = eeState_w[:, 3:7]
        self.eeVel[env_ids] = eeState_w[:, 7:]

        # block State
        blockState_w = self._block.data.body_link_state_w[env_ids, 0].clone()
        self.block_pos_w[env_ids] = blockState_w[:, :3]
        self.block_quat_w[env_ids] = blockState_w[:, 3:7]
        self.block_vel[env_ids] = blockState_w[:, 7:]

        # joint state
        self.joint_pos[env_ids] = self._robot.data.joint_pos[env_ids].clone()
        self.joint_vel[env_ids] = self._robot.data.joint_vel[env_ids].clone()

        self.contactForce[env_ids] = self._contactSensor.data.force_matrix_w[env_ids].squeeze()

        if self.cfg.planar:
            self.joint_pos_fifo = torch.cat([self.joint_pos_fifo[:,1:,:], self.joint_q[:,1:4].reshape(-1, 1, 3)], dim=1)
            self.joint_vel_fifo = torch.cat([self.joint_vel_fifo[:,1:,:], self.joint_qd[:,1:4].reshape(-1, 1, 3)], dim=1)
        else:
            self.joint_pos_fifo = torch.cat([self.joint_pos_fifo[:,1:,:], self.joint_q.reshape(-1, 1, 6)], dim=1)
            self.joint_vel_fifo = torch.cat([self.joint_vel_fifo[:,1:,:], self.joint_qd.reshape(-1, 1, 6)], dim=1)

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        """Object Position"""
        jointAcc_norm = torch.linalg.vector_norm(self.cmd_acc, dim=1, ord=1)
        jointAcc_rate_penalty = torch.linalg.vector_norm((self.cmd_acc - self.last_cmd_acc), dim=1, ord=1)
        self.last_cmd_acc = self.cmd_acc.clone()
        jointVel_norm = torch.linalg.vector_norm(self.joint_vel, dim=1, ord=2)


        target_pos =  self.target_pos + self._robot.data.root_link_state_w[:, :3]
        block_pos = self._block.data.root_link_state_w[:, :3]
        block_pos[:, 2] -= self.real_assets_sizes[:, 2] / 2
        block_pos_error = torch.linalg.vector_norm(block_pos- target_pos, dim=1, ord=2)


        is_success = (block_pos_error < 0.25) 
        success_rew =  is_success.float() * (logistic_kernel(-(block_pos_error), std=0.1))

        reg_scale = success_rew.mean()

        # Compute direction to target
        direction_to_target = target_pos - self.block_pos_w
        direction_to_target_norm = direction_to_target / (torch.linalg.vector_norm(direction_to_target, dim=1, keepdim=True) + 1e-6)

        # Block velocity
        block_vel = self.block_vel[:, :3]  # Assuming first 3 are linear velocity

        # Velocity towards target (dot product)
        block_vel_norm = block_vel[:,:2]  / (torch.linalg.vector_norm(block_vel[:,:2] , dim=1, keepdim=True) + 1e-6)
        velocity_towards_target = torch.sum(block_vel_norm * direction_to_target_norm[:,:2], dim=1) # xy

        contactForce_norm = torch.linalg.vector_norm(self.contactForce, dim=1, ord=2)
        post_throw = (contactForce_norm == 0.0) 
        double_post_throw = (self.steps_since_postThrow > 0) & (~post_throw)

        self.steps_since_postThrow[~post_throw] = 0
        self.steps_since_postThrow[post_throw] += 1

        eePos_x = self.eePos_w[:, 0] - self.robot_root_pos_w[:, 0]
        not_far_ee_x = eePos_x > -0.5
        
        rewards = (

            self.cfg.live_reward*post_throw.float()
            - self.cfg.double_penalty * double_post_throw.float()

            - self.cfg.densePos_rewScale *block_pos_error
            + self.cfg.denseVel_rewScale * velocity_towards_target

            + self.cfg.sparse_rewScale * success_rew

            - self.cfg.reg_rewScale * jointAcc_norm* reg_scale
            - self.cfg.reg_rewScale * jointAcc_rate_penalty*reg_scale
            #-self.cfg.reg_rewScale * jointVel_norm* reg_scale*post_throw.float()*10


        )



        self.extras["log"] = {
            "scale": self.scale,
            "post_throw": post_throw,

            "BlockError": block_pos_error,
            "velocity_towards_target": velocity_towards_target,
            
            "is_success": is_success & not_far_ee_x,
            "success_rew": success_rew,

            "JointAcc": self.cmd_acc,
            "jointAcc_norm": jointAcc_norm,
            "jointVel_norm": jointVel_norm,

            "block_state": torch.cat((self.block_pos_w- self._robot.data.root_link_state_w[:, :3], self.block_quat_w, self.block_vel, post_throw.reshape(-1,1)), dim=-1),
            "ee_state": torch.cat((self.eePos_w, self.eeQuat_w, self.eeVel), dim=-1),
            "joint_state": torch.cat((self.joint_pos, self.joint_vel), dim=-1),
            "target_pos": self.target_pos,
            "ObjectModel": torch.cat((self.block_mass, self.block_staticFriction, self.block_dynamicFriction, self.block_restitution, self.real_assets_sizes, self.block_com, self.x_shift.reshape(-1,1)), dim=-1),
        }

        return rewards



    def init_state_buffers(self):
        """Initialize the state buffers for the environment"""
        self.init_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.init_joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.block_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.block_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.block_vel = torch.zeros((self.num_envs, 6), device=self.device)

        self.eePos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.eeQuat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.eeVel = torch.zeros((self.num_envs, 6), device=self.device)

        self.joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        if self.cfg.planar:
            self.actions_fifo = torch.zeros((self.num_envs, self.cfg.histLen, 3), device=self.device)
            self.joint_pos_fifo = torch.zeros((self.num_envs, self.cfg.histLen, 3), device=self.device)
            self.joint_vel_fifo = torch.zeros((self.num_envs, self.cfg.histLen, 3), device=self.device)
        else:
            self.actions_fifo = torch.zeros((self.num_envs, self.cfg.histLen, 6), device=self.device)
            self.joint_pos_fifo = torch.zeros((self.num_envs, self.cfg.histLen, 6), device=self.device)
            self.joint_vel_fifo = torch.zeros((self.num_envs, self.cfg.histLen, 6), device=self.device)

        self.last_action = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.actions = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.joint_q = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.joint_qd = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.cmd_acc = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.last_cmd_acc = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self.contactForce = torch.zeros((self.num_envs, 3), device=self.device)
        self.steps_since_postThrow = torch.zeros((self.num_envs,), device=self.device)
        self.bin_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.x_shift = torch.zeros((self.num_envs,), device=self.device)

    def update_curriculum(self):
        avg_ep_length = 1.1 * self.episode_length_buf.float().mean() / (self.cfg.max_Steps / 2)
        scale = min(avg_ep_length, 1)
        scale = 0.1 + 0.9*scale
        self.scale = scale 
        self.cfg.densePos_rewScale = 1.0 * self.scale
        self.cfg.denseVel_rewScale = 10.0 * self.scale
        self.cfg.sparse_rewScale = 100.0 * self.scale
        self.cfg.reg_rewScale = 1e-1* self.scale

        
        #if self.cfg.training:
        #    self.cfg.action_scale = 40.0*self.scale
    def sample_env_state(self, env_ids):
        self.init_joint_pos[env_ids] = self._sample_init_joint(self._robot.data.default_joint_pos[env_ids])

        self._robot.write_joint_state_to_sim(self.init_joint_pos[env_ids], self.init_joint_vel[env_ids], env_ids=env_ids)


        block_pose = self._robot.data.body_link_state_w[env_ids, self.tray_link_idx][:, :7].clone()
        block_pose[:, 2] += self.real_assets_sizes[env_ids, 2] / 2 #+ 0.1 # add a small offset to the block position
        block_vel_envIds = torch.zeros((len(env_ids), 6), device=self.device)

        
        #radius = 0.9*torch.rand(len(env_ids), device=self.device)*0.1
        #self.x_shift[env_ids] = 0.1*(torch.rand(len(env_ids), device=self.device)-0.5)*2
        #theta = 2*math.pi*torch.rand(len(env_ids), device=self.device)
        #theta = torch.zeros_like(theta)
        block_pose[:, 0] += self.x_shift[env_ids]#radius * torch.cos(theta)
        #block_pose[:, 1] += radius * torch.sin(theta)


        self._block.write_root_pose_to_sim(block_pose, env_ids=env_ids)
        self._block.write_root_velocity_to_sim(block_vel_envIds, env_ids=env_ids)
        
        target_angle = self.init_joint_pos[env_ids, 0] #(torch.rand_like(self.init_joint_pos[env_ids, 0])-0.5) * math.pi/2
        #target_angle = (torch.rand_like(self.init_joint_pos[env_ids, 0])-0.5) * math.pi/2

        target_radius = 2.5 + (torch.rand_like(self.target_pos[env_ids, 0])-0.5) * 2 * 1.0
        #target_radius = 4 + torch.rand_like(self.target_pos[env_ids, 0])*0.5

        #target_radius = torch.ones_like(target_radius)*3.5

        target_x = target_radius * torch.cos(target_angle) + 0.133 * torch.sin(target_angle)
        target_y = target_radius * torch.sin(target_angle) + 0.133 * torch.cos(target_angle)

        target_z = (0.1 + (torch.rand_like(self.target_pos[env_ids, 2])-0.5) * 2 * 1.0)  # 0.8 is robot stand height
        #target_z = 1.6 + torch.rand_like(self.target_pos[env_ids, 2]) * 0.5  # 0.8 is robot stand height

        #target_z = torch.ones_like(target_y)*(-0.88+0.15)
        #target_z = torch.ones_like(target_y)*(-0.88+0.8)
        #target_z = torch.ones_like(target_y)*(-0.88+1.8)


        self.target_pos[env_ids, 0] = target_x
        self.target_pos[env_ids, 1] = target_y
        self.target_pos[env_ids, 2] = target_z

        self.bin_pose[env_ids] = self._robot.data.root_link_state_w[env_ids, :7].clone()
        self.bin_pose[env_ids,:3] += self.target_pos[env_ids].clone()
        self.bin_pose[env_ids, 2] -= -0.005

        self._bin.write_root_pose_to_sim(self.bin_pose[env_ids], env_ids=env_ids)
        
    def _sample_init_joint(self, init_joint_pos):
        """Sample joints ensuring initial horizontal orientation and randomize the position"""
        n = init_joint_pos.shape[0]

        #init_joint_pos[:, 0] = math.pi/2 * (torch.rand_like(init_joint_pos[:, 0]) - 0.5)
        
        # Sample random angle between -45 and 45 degrees
        angle = math.pi / 2 * (torch.rand_like(init_joint_pos[:, 0]) - 0.5)
        coeff1 = 2 * torch.rand_like(init_joint_pos[:, 0]) - 1 # why not  torch.rand_like(init_joint_pos[:, 0])
        coeff2 = 2 * torch.rand_like(init_joint_pos[:, 0]) - 1
        coeff3 = coeff1 + coeff2
        init_joint_pos[:, 1] += coeff1 * angle
        init_joint_pos[:, 3] += coeff2 * angle
        init_joint_pos[:, 2] -= coeff3 * angle

        return init_joint_pos
    

    

@torch.jit.script
def logistic_kernel(x: torch.Tensor, std: float):
    a = 1 / std
    return (2.0) / (torch.exp(x * a) + torch.exp(-x * a))

