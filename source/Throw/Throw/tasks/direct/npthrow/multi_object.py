# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn multiple objects in multiple environments.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/demos/multi_object.py --num_envs 2048

"""

from __future__ import annotations


"""Rest everything follows."""
import torch
import random
import re
import traceback
from dataclasses import MISSING

import carb
import isaacsim.core.utils.prims as prim_utils
import omni.usd
from pxr import Gf, Sdf, Semantics, Usd, UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass


def spawn_multi_object_randomly(
    prim_path: str,
    cfg: MultiAssetCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    source_prim_paths = [f"/World/envs/env_{i}" for i in range(16)]

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # manually clone prims if the source prim path is a regex expression
    for i, prim_path in enumerate(prim_paths):
        # randomly select an asset configuration
        asset_cfg = cfg.assets_cfg[i]  # random.choice(cfg.assets_cfg)
        # spawn single instance
        prim = asset_cfg.func(prim_path, asset_cfg, translation, orientation)
        # set the prim visibility
        if hasattr(asset_cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if asset_cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(asset_cfg, "semantic_tags") and asset_cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in asset_cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(asset_cfg, "activate_contact_sensors") and asset_cfg.activate_contact_sensors:
            sim_utils.activate_contact_sensors(prim_path, asset_cfg.activate_contact_sensors)

    # return the prim
    return prim


def spawn_multi_object_randomly_sdf(
    prim_path: str,
    cfg: MultiAssetCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage

    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # print("source_prim_paths", source_prim_paths)
    # source_prim_paths = [f"/World/envs/env_{i}" for i in range(24)]
    # spawn everything first in a "Dataset" prim
    prim_utils.create_prim("/World/Dataset", "Scope")
    proto_prim_paths = list()
    for index, asset_cfg in enumerate(cfg.assets_cfg):
        # spawn single instance
        proto_prim_path = f"/World/Dataset/Object_{index:02d}"
        prim = asset_cfg.func(proto_prim_path, asset_cfg)
        # save the proto prim path
        proto_prim_paths.append(proto_prim_path)
        # set the prim visibility
        if hasattr(asset_cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if asset_cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(asset_cfg, "semantic_tags") and asset_cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in asset_cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(asset_cfg, "activate_contact_sensors") and asset_cfg.activate_contact_sensors:
            sim_utils.activate_contact_sensors(proto_prim_path, asset_cfg.activate_contact_sensors)

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # convert orientation ordering (wxyz to xyzw)
    orientation = (orientation[1], orientation[2], orientation[3], orientation[0])

    assets_sizes = torch.tensor(cfg.assets_sizes).to("cuda")
    assets_colors = torch.tensor(cfg.assets_colors).to("cuda")
    # manually clone prims if the source prim path is a regex expression
    # print("proto_prim_paths", proto_prim_paths)
    with Sdf.ChangeBlock():
        for i, prim_path in enumerate(prim_paths):
            # spawn single instance
            env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            # randomly select an asset configuration
            proto_path = random.choice(proto_prim_paths)
            # inherit the proto prim
            # env_spec.inheritPathList.Prepend(Sdf.Path(proto_path))
            Sdf.CopySpec(env_spec.layer, Sdf.Path(proto_path), env_spec.layer, Sdf.Path(prim_path))
            # set the translation and orientation
            _ = UsdGeom.Xform(stage.GetPrimAtPath(proto_path)).GetPrim().GetPrimStack()

            translate_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")
            if translate_spec is None:
                translate_spec = Sdf.AttributeSpec(env_spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate_spec.default = Gf.Vec3d(*translation)

            orient_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:orient")
            if orient_spec is None:
                orient_spec = Sdf.AttributeSpec(env_spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
            orient_spec.default = Gf.Quatd(*orientation)

            op_order_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
            if op_order_spec is None:
                op_order_spec = Sdf.AttributeSpec(env_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray)
            op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

            # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            # Note: Just need to acquire the right attribute about the property you want to set
            # Here is an example on setting color randomly
            color_spec = env_spec.GetAttributeAtPath(prim_path + "/geometry/material/Shader.inputs:diffuseColor")
            color = [float(c) for c in assets_colors[i]]
            color_spec.default = Gf.Vec3f(*color)

            asset_size = assets_sizes[i]
            
            size_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            width = float(asset_size[0])
            length= float(asset_size[1])
            height = float(asset_size[2])

            if size_spec is None:
                size_spec = Sdf.AttributeSpec(env_spec, "xformOp:scale", Sdf.ValueTypeNames.Double3)
            size_spec.default = Gf.Vec3d(width, length, height)

    # delete the dataset prim after spawning
    prim_utils.delete_prim("/World/Dataset")

    # return the prim
    return prim_utils.get_prim_at_path(prim_paths[0])


@configclass
class MultiAssetCfg(sim_utils.SpawnerCfg):
    """Configuration parameters for loading multiple assets randomly."""

    # Uncomment this one: 45 seconds for 2048 envs
    # func: sim_utils.SpawnerCfg.func = spawn_multi_object_randomly
    # Uncomment this one: 2.15 seconds for 2048 envs
    func: sim_utils.SpawnerCfg.func = spawn_multi_object_randomly_sdf

    assets_cfg: list[sim_utils.SpawnerCfg] = MISSING
    """List of asset configurations to spawn."""

    assets_sizes: torch.Tensor = MISSING
    """List of asset sizes to spawn."""
    assets_colors: torch.Tensor = MISSING
    """List of asset colors to spawn."""


@configclass
class MultiObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=MultiAssetCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.1, 0.1, 0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="average",
                        restitution_combine_mode="average",
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                )
            ]
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
    )


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    rigid_object = scene["object"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            root_state = rigid_object.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            rigid_object.write_root_state_to_sim(root_state)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cuda")
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = MultiObjectSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0)
    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
