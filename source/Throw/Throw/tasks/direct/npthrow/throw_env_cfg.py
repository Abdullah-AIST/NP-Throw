# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, GaussianNoiseCfg, NoiseModelCfg
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
import Throw.tasks.direct.npthrow.mdp as mdp
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from gymnasium import spaces

from .multi_object import MultiAssetCfg

import os
import math
import torch
from torch.distributions import Beta


def sample_beta(n: int, x_min:float, x_max:float, alpha=1, beta=1, isLog=False) -> torch.Tensor:
    """Sample from a beta distribution and scale to a given range."""
    # Sample from a beta distribution
    beta_dist = Beta(alpha, beta)
    samples = beta_dist.sample((n,1))
    # Scale the samples to the desired range
    if isLog:
        x_max = math.log(x_max)
        x_min = math.log(x_min)
    scaled_samples = samples * (x_max - x_min) + x_min
    if isLog: scaled_samples = torch.exp(scaled_samples)  # Apply exponential if log scaling is desired
    return scaled_samples



@configclass
class EventCfg:
    """Configuration for randomization."""

    
    # wood 0.3
    # lego 0.3 -- 0.4 -- 
    # legoJar 0.25
    # chips 0.3 -- 0.35-0.3
    # bleach 0.2
    #largeBox 0.2

    # -- object
    # uncomment to enable randomization
    #object_physics_material = EventTerm(
    #    func=mdp.randomize_rigid_body_material,
    #    min_step_count_between_reset=0, # reset every episode
    #    mode="reset",
    #    params={
    #        "asset_cfg": SceneEntityCfg("block"),
    #        "static_friction_range": (0.2, 1.0),      # 0.1, 1.0  #0.1 friction is safe for most objects but slower  #0.2, 0.15, 0.0 --- 0.3, 0.25, 0.0
    #        "dynamic_friction_range": (0.2, 1.0),      # 0.1, 1.0
    #        "restitution_range": (0.0, 0.5),           # (0.0, 0.5),
    #        "num_buckets": 500,
    #        "make_consistent": True,
    #    },
    #)
    #object_abs_mass = EventTerm(
    #    func=mdp.randomize_rigid_body_mass,
    #    min_step_count_between_reset=0,
    #    mode="reset",
    #    params={
    #        "asset_cfg": SceneEntityCfg("block"),
    #        "mass_distribution_params": (0.1, 1.0), # (0.05, 0.5), #rubic,chips: 0.1, liquid:0.2,
    #        "operation": "abs",
    #        "distribution": "uniform",
    #        "recompute_inertia": True, 
    #    },
    #)
#

    #object_com = EventTerm(
    #    func=mdp.randomize_rigid_body_COM,
    #    mode="reset",                       #"startup","reset"
    #    params={
    #        "asset_cfg": SceneEntityCfg("block"),
    #        "limit": 0.9,  # in each axis, as a fraction of the half-size of the object
    #    },
    #)


@configclass
class ThrowEnvCfg(DirectRLEnvCfg):
    AccRate = 0.1             #0.5 seems to streak a balance``
    action_scale = 40.0

    training = False
    evaluating = (not training) 

    RandCOM = False
    RandObjPos = False

    planar = True

    histLen = 10 #action history length

    sim_Freq = 120
    ctrl_Freq = 20 #20*1.5
    decimation = int(sim_Freq / ctrl_Freq)
    nTrajsPerEpisode = 1 # if not evaluating else 1
    TrajTime = 3.2 #if (not training or  RandCOM or RandObjPos) else 3.0
    episode_length_s = nTrajsPerEpisode* TrajTime #if not evaluating else 3 # 48*3/60=2.4
    action_space = 3 if planar else 6 #spaces.Box(-2, 2, shape=(6,))
    observation_space =  8 + (9 + 9*(not planar))*histLen +3 #+1#+ 9 
    state_space =   38 + (9+9*(not planar))*histLen + 3 + 1 +3*RandCOM #+1 # 31 + 1 + 2 + 6 + 3
    maxEp_steps = int(TrajTime*ctrl_Freq) # 48*3=144
    max_Steps = int(episode_length_s * ctrl_Freq)  # freq(120)/decimation(2)
    
    # num of instances
    n_training = 4096
    n_eval = 4096

    
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, #1/120
        render_interval=decimation,
        #disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        use_fabric=True, #set to False for visual debugging
    )

    # scene
    spacing = 4
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=spacing, replicate_physics=False)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="./assets/ur5eTray.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -math.pi / 4,
                "elbow_joint": math.pi / 2,  # 1.712,
                "wrist_1_joint": -math.pi / 4,
                "wrist_2_joint": math.pi / 2,
                "wrist_3_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.88),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.4,
                damping=40000,
               )
        },
    )


    stand = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Stand",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.88),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.44), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    bin = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Bin",
        spawn=sim_utils.UsdFileCfg(
            usd_path="./assets/rectBin.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, -0.005 +1.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )


    custom_ground = AssetBaseCfg(
        prim_path="/World/envs/env_.*/CustomGround",
        spawn=sim_utils.CuboidCfg(
            size=(spacing-0.1, spacing-0.1, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),

            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.1)),
            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.25), rot=(1.0, 0.0, 0.0, 0.0)),
    )


    assets_sizes = torch.empty((n_training + n_eval, 3), device="cuda")

  

    assets_sizes[:, 0] = sample_beta(n_training + n_eval, 0.05,  0.2, alpha=1.0, beta=1.0,isLog=False).squeeze()     # encourage taller objects up to 0.6
    assets_sizes[:, 1] = sample_beta(n_training + n_eval, 0.05,  0.2, alpha=1.0, beta=1.0,isLog=False).squeeze()     # encourage thinner objects up to 0.2 ratio (0.2*0.6 = 0.12)
    assets_sizes[:, 2] = sample_beta(n_training + n_eval, 0.05,  0.25, alpha=1.0, beta=1.0,isLog=False).squeeze()       # Uniform distribution for the width/length ratio

    assets_colors = torch.rand((n_training + n_eval, 3), device="cuda")
    assets_frictions = torch.rand((n_training + n_eval, 1), device="cuda") * 0.9 + 0.1


    if training:
        assets_sizes = assets_sizes[:n_training]
        assets_colors = assets_colors[:n_training]
        assets_frictions = assets_frictions[:n_training]
    else:
        assets_sizes = assets_sizes[n_training:]
        assets_colors = assets_colors[n_training:]
        assets_frictions = assets_frictions[n_training:]


    # use target="block" for random sized blocks
    target = "woodBlock"  #"cylinder", "block", "customBlock", "woodBlock", "chips", "crackerBox", "mustardBottle", "powerDrill", "bleachCleanser"

    YCB_tragets = {
                    "chips": [0.075, 0.075, 0.250],
                    "woodBlock2": [0.2, 0.085, 0.085],
                    "largeBox": [0.2,0.2,0.28],
                    "rubiks": [0.06,0.06,0.06],
                    "liquid": [0.06,0.06,0.17],

                    "sensBlock": [0.1, 0.1, 0.1],
                    "bestBlock": [0.2,0.1,0.05],

                    "dexCube": [0.1, 0.1, 0.1],
                    #"crackerBox": [0.06, 0.158, 0.21], 
                    "crackerBox": [0.158, 0.06, 0.21], 

                    "chips": [0.075, 0.075, 0.250],
                    #"mustardBottle": [0.058, 0.095, 0.190], 
                    "mustardBottle": [0.095, 0.058, 0.190],  # rotated
                    "masterChefCan": [0.102, 0.102, 0.139],
                    "soupCan": [0.066, 0.066, 0.101],
                    "spam": [0.050, 0.097, 0.082],
                    "sugarBox": [0.038, 0.089, 0.175],
                    "tunaCan": [0.085, 0.085, 0.033],
                    "gelatin": [0.028, 0.085, 0.073],
                    "pudding": [0.035, 0.110, 0.089],
                    "apple": [0.075, 0.075, 0.075],
                    "pitcher": [0.108,0.108,0.235],
                    "bowl": [0.159, 0.159, 0.053],
                    "mug": [0.080, 0.080, 0.082],
                    #"bleachCleanser": [0.065, 0.098, 0.250],
                    "bleachCleanser": [0.098, 0.065, 0.250], # rotated

                    #"powerDrill": [0.05, 0.15, 0.184],
                    "powerDrill": [0.15, 0.05, 0.184], # rotated

                    "woodBlock": [0.2, 0.085, 0.085],

                       }
    if target == "cylinder":
        assets_sizes[:,1] = assets_sizes[:,0]

    if target in YCB_tragets.keys():
        print("Evaluating YCB object: ", target)
        assets_sizes = torch.ones_like(assets_sizes)
        assets_sizes[:,0] = assets_sizes[:,0]*YCB_tragets[target][0]
        assets_sizes[:,1] = assets_sizes[:,1]*YCB_tragets[target][1]
        assets_sizes[:,2] = assets_sizes[:,2]*YCB_tragets[target][2]

        
    
    #change from torch to list
    assets_sizes = assets_sizes.tolist()
    assets_colors = assets_colors.tolist()
    assets_frictions = assets_frictions.tolist()

    if target == "block" or target == "customBlock":
        block = RigidObjectCfg(
            prim_path="/World/envs/env_.*/block",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0], lin_vel=[0, 0, 0], ang_vel=[0, 0, 0]
            ),
            spawn=MultiAssetCfg(
                assets_cfg=[
                    sim_utils.CuboidCfg( #CuboidCfg, CylinderCfg, SphereCfg
                        #height = 1,
                        #radius=0.5,
                        size=(1, 1, 1),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(density=100),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
                        physics_material=sim_utils.RigidBodyMaterialCfg(
                            friction_combine_mode="multiply",
                            restitution_combine_mode="multiply",
                            static_friction=0.35,
                            dynamic_friction=0.3,
                            restitution=0.0,
                        ),
                    )
                ],
                assets_sizes=assets_sizes,
                assets_colors=assets_colors,
            ),
        )
    elif target == "cylinder":
        block = RigidObjectCfg(
            prim_path="/World/envs/env_.*/block",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0], lin_vel=[0, 0, 0], ang_vel=[0, 0, 0]
            ),
            spawn=MultiAssetCfg(
                assets_cfg=[
                    sim_utils.CylinderCfg( #CuboidCfg, CylinderCfg, SphereCfg
                        height = 1,
                        radius=0.5,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(density=100),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
                        physics_material=sim_utils.RigidBodyMaterialCfg(
                            friction_combine_mode="multiply",
                            restitution_combine_mode="multiply",
                            static_friction=0.3,
                            dynamic_friction=0.3,
                            restitution=0.0,
                        ),
                    )
                ],
                assets_sizes=assets_sizes,
                assets_colors=assets_colors,
            ),
        )
    else:
        block = RigidObjectCfg(
            prim_path="/World/envs/env_.*/block",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0], lin_vel=[0, 0, 0], ang_vel=[0, 0, 0]
            ),
            spawn=sim_utils.UsdFileCfg(usd_path=f"./assets/{target}2.usd",) 

        )
    

    
    
    
    
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/tray",
        update_period=1/ctrl_Freq,  # 1/40
        filter_prim_paths_expr=["/World/envs/env_.*/block"],
        #track_air_time=True,
    )

    frame_object_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/frame_marker",
        markers={
             "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
        ),
        },
    )

    goal_object_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/frame_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.1, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0))
            ),
        },
    )

    # reward scales
    # survival/completion
    live_reward = 1.0
    double_penalty = 10

    densePos_rewScale = 1
    denseVel_rewScale = 10

    sparse_rewScale = 100

    reg_rewScale = 0.1


