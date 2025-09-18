# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

#import numpy as np
#import cvxpy as cp
import carb
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_dof_velocity_limit_override(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set the velocity limit specified in the cfg as a usd property. Otherwise it is not respected"""
    # Discard env_ids and apply the changes to all environments
    env_ids_override = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    for k in asset.cfg.actuators.keys():
        velocity_limit = torch.tensor([asset.cfg.actuators[k].velocity_limit], device=env.device).repeat(env.num_envs, 1)
        #sset.root_physx_view.set_dof_max_velocities(velocity_limit, env_ids_override)
        asset.write_joint_velocity_limit_to_sim(velocity_limit, env_ids_override)

def randomize_rigid_body_COM(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
):
    """Randomize the mass of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    If the ``recompute_inertia`` flag is set to ``True``, the function recomputes the inertia tensor of the bodies
    after setting the mass. This is useful when the mass is changed significantly, as the inertia tensor depends
    on the mass. It assumes the body is a uniform density object. If the body is not a uniform density object,
    the inertia tensor may not be accurate.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get asset size
    #asset_size =  asset.root_physx_view.get_local_scales()
    
    # get the current masses of the bodies (num_assets, num_bodies)
    asset_coms = asset.root_physx_view.get_coms()
    asset_sizes = torch.tensor(env.cfg.assets_sizes, device="cpu")[env_ids]

    range_low = torch.tensor(3*[-0.50], device="cpu")
    range_high = torch.tensor(3*[0.50], device="cpu")
    rand_samples = math_utils.sample_uniform(
        range_low, range_high, (len(env_ids), len(body_ids), 3), device="cpu"
    ).squeeze()

    asset_coms_new = asset_sizes * rand_samples/2

    # Randomize the com in range
    asset_coms[env_ids, :3] = asset_coms_new

    # Recompute inertia for each body
    masses = asset.root_physx_view.get_masses().clone().squeeze()
    asset_inertia_new = compute_cuboid_inertia_with_offset_com(
        masses[env_ids],
        asset_sizes,
        asset_coms_new
    )
    
    #asset_inertia_new = []
    #default_inertia = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
    #for i in range(masses.shape[0]):
    #    IC = max_min_eig_inertia(asset_sizes[i], asset_coms_new[i], diag=True)*float(masses[i])
    #    inertia = default_inertia.copy()
    #    inertia[0] = IC[0]
    #    inertia[4] = IC[1]
    #    inertia[8] = IC[2]
    #    asset_inertia_new.append(inertia)
    #asset_inertia_new = torch.tensor(asset_inertia_new, device="cpu").squeeze()
    
    inertias = asset.root_physx_view.get_inertias().clone()
    #print("Inertia before: ", inertias.shape, inertias[0])
    #print("Inertia after: ", asset_inertia_new.shape, asset_inertia_new[0])
    #print("Inertia before: ", inertias.shape, inertias[0])
    #print("Inertia after: ", asset_inertia_new.shape, asset_inertia_new[0])
    #print("Environment IDs: ", len(env_ids))
    inertias[env_ids, :] = asset_inertia_new
    # Set the new coms
    asset.root_physx_view.set_coms(asset_coms, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)

"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device) # (n_dim_0, n_dim_1) or (n_dim_0, 1)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def compute_cuboid_inertia_with_offset_com(
    mass: float,
    size: tuple[float, float, float],
    com_offset: tuple[float, float, float]
) -> torch.Tensor:
    """Compute the inertia matrix of a cuboid with offset center of mass.
    
    Args:
        mass: Mass of the cuboid
        size: (length, width, height) of the cuboid
        com_offset: (x, y, z) offset of center of mass from geometric center
        
    Returns:
        3x3 inertia tensor matrix
    """
    l, w, h = size[:,0], size[:,1], size[:,2]
    dx, dy, dz = com_offset[:,0], com_offset[:,1], com_offset[:,2]
    # First compute inertia about geometric center
    Ixx = (mass/12.0) * (w*w + h*h)  # About x-axis
    Iyy = (mass/12.0) * (l*l + h*h)  # About y-axis 
    Izz = (mass/12.0) * (l*l + w*w)  # About z-axis
    
    # Use parallel axis theorem to get inertia about offset COM
    # I = I_c + m*d^2 where d is perpendicular distance to axis
    Ixx += mass * (dy*dy + dz*dz)
    Iyy += mass * (dx*dx + dz*dz) 
    Izz += mass * (dx*dx + dy*dy)
    
    # Products of inertia
    Ixy = -mass * dx * dy  # xy plane
    Iyz = -mass * dy * dz  # yz plane
    Ixz = -mass * dx * dz  # xz plane
    
    # Construct 3x3 inertia tensor
    #inertia = torch.stack([
    #    torch.stack([Ixx, -Ixy, -Ixz], dim=-1),
    #    torch.stack([-Ixy, Iyy, -Iyz], dim=-1),
    #    torch.stack([-Ixz, -Iyz, Izz], dim=-1)
    #], dim=-2)
    inertia = torch.stack([Ixx, -Ixy, -Ixz, -Ixy, Iyy, -Iyz, -Ixz, -Iyz, Izz], dim=-1)
    return inertia


def max_min_eig_inertia(boxSize, com, diag=True):
    """Find the inertia matrix about the CoM with the maximum smallest eigenvalue.

    The masses are places at the vertices of the box.
    """
    x, y, z = boxSize[0], boxSize[1], boxSize[2]
    vertices = 0.5*np.array(
        [
            [x, y, z],
            [x, y, -z],
            [x, -y, z],
            [x, -y, -z],
            [-x, y, z],
            [-x, y, -z],
            [-x, -y, z],
            [-x, -y, -z],
        ]
    )
    μs = cp.Variable(8)
    vs = [np.append(v, 1) for v in vertices]
    J = cp.sum([μ * np.outer(v, v) for μ, v in zip(μs, vs)])

    Hc = J[:3, :3] - np.outer(com, com)  # Hc is about the CoM
    λ = cp.Variable(1)
    objective = cp.Maximize(λ)
    drip_constraints = [
        Hc >> 0,
        cp.sum(μs) == 1,
        μs >= 0,
        λ >= 0,
    ]

    # if diag=True, only optimize over the diagonal of I
    if diag:
        Ic = cp.Variable(3)
        constraints = [
            λ * np.ones(3) <= Ic,
            cp.diag(Ic) == cp.trace(Hc) * np.eye(3) - Hc,
        ] + drip_constraints
    else:
        Ic = cp.Variable((3, 3))
        constraints = [
            λ * np.eye(3) << Ic,
            Ic == cp.trace(Hc) * np.eye(3) - Hc,
        ] + drip_constraints

    problem = cp.Problem(objective, constraints)
    problem.solve(cp.CLARABEL)
    # print(Ic.value)
    # print(problem.value)

    return Ic.value