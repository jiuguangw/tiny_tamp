from __future__ import annotations

import math
import random
from typing import Callable, List

import numpy as np

import tiny_tamp.pb_utils as pbu
from tiny_tamp.inverse_kinematics.franka_panda.ik import PANDA_INFO
from tiny_tamp.inverse_kinematics.ikfast import (
    closest_inverse_kinematics,
    get_ik_joints,
    ikfast_inverse_kinematics,
)
from tiny_tamp.structs import (
    COLLISION_DISTANCE,
    COLLISION_EPSILON,
    MAX_IK_DISTANCE,
    MAX_IK_TIME,
    SELF_COLLISIONS,
    ActivateGrasp,
    Attachment,
    Conf,
    DeactivateGrasp,
    Grasp,
    Sequence,
    SimulatorInstance,
    Trajectory,
    WorldBelief,
)


def get_plan_motion_fn(
    sim: SimulatorInstance, environment: List[int] = [], debug: bool = False
):
    """Plan a motion from one configuration to another."""

    def fn(q1: Conf, q2: Conf, attachments: List[Attachment] = []):
        obstacles = list(environment)
        attached = {attachment.child for attachment in attachments}
        obstacles = set(obstacles) - attached
        q1.assign(sim)

        resolutions = math.radians(10) * np.ones(len(q2.joints))

        path = pbu.plan_joint_motion(
            sim.robot,
            q2.joints,
            q2.positions,
            resolutions=resolutions,
            obstacles=obstacles,
            attachments=attachments,
            self_collisions=SELF_COLLISIONS,
            max_distance=COLLISION_DISTANCE,
            restarts=1,
            iterations=5,
            smooth=100,
            debug=debug,
            disable_collisions=False,
            client=sim.client,
        )

        if path is None:
            return None

        return Trajectory(sim.robot, q2.joints, path)

    return fn


def plan_workspace_motion(
    sim: SimulatorInstance,
    tool_waypoints,
    attachment=None,
    obstacles=[],
    max_attempts=2,
    debug=False,
):
    """Return a joint path that moves the tool along the tool waypoints.

    This is useful if you want to move the gripper in a straight line
    path.
    """

    assert tool_waypoints

    tool_link = sim.tool_link
    ik_joints = get_ik_joints(sim.robot, PANDA_INFO, tool_link, client=sim.client)
    fixed_joints = set(ik_joints) - set(sim.get_group_joints(sim.arm_group))
    arm_joints = [j for j in ik_joints if j not in fixed_joints]
    extract_arm_conf = lambda q: [
        p for j, p in pbu.safe_zip(ik_joints, q) if j not in fixed_joints
    ]

    parts = [sim.robot] + ([] if attachment is None else [attachment.child])
    collision_fn = pbu.get_collision_fn(
        sim.robot,
        arm_joints,
        obstacles=obstacles,
        attachments=[],
        self_collisions=SELF_COLLISIONS,
        disable_collisions=False,
        client=sim.client,
    )

    for attempts in range(max_attempts):
        for arm_conf in ikfast_inverse_kinematics(
            sim.robot,
            PANDA_INFO,
            tool_link,
            tool_waypoints[0],
            fixed_joints=fixed_joints,
            max_attempts=5,
            max_time=np.inf,
            max_distance=None,
            use_halton=False,
            client=sim.client,
        ):
            arm_conf = extract_arm_conf(arm_conf)

            if collision_fn(arm_conf):
                continue

            arm_waypoints = [arm_conf]
            for tool_pose in tool_waypoints[1:]:
                arm_conf = next(
                    closest_inverse_kinematics(
                        sim.robot,
                        PANDA_INFO,
                        tool_link,
                        tool_pose,
                        fixed_joints=fixed_joints,
                        max_candidates=np.inf,
                        max_time=MAX_IK_TIME,
                        max_distance=MAX_IK_DISTANCE,
                        verbose=False,
                        client=sim.client,
                    ),
                    None,
                )
                if arm_conf is None:
                    break

                arm_conf = extract_arm_conf(arm_conf)
                if collision_fn(arm_conf):
                    if debug:
                        pbu.wait_if_gui(client=sim.client)
                    continue
                arm_waypoints.append(arm_conf)
            else:
                pbu.set_joint_positions(
                    sim.robot, arm_joints, arm_waypoints[-1], client=sim.client
                )
                if attachment is not None:
                    attachment.assign(sim)
                if any(
                    pbu.pairwise_collisions(
                        part,
                        obstacles,
                        max_distance=(COLLISION_DISTANCE + COLLISION_EPSILON),
                        client=sim.client,
                    )
                    for part in parts
                ):
                    if debug:
                        pbu.wait_if_gui(client=sim.client)
                    continue
                arm_path = pbu.interpolate_joint_waypoints(
                    sim.robot, arm_joints, arm_waypoints, client=sim.client
                )

                if any(collision_fn(q) for q in arm_path):
                    if debug:
                        pbu.wait_if_gui(client=sim.client)
                    continue

                print(
                    "Found path with {} waypoints and {} configurations after {} attempts".format(
                        len(arm_waypoints), len(arm_path), attempts + 1
                    )
                )

                return arm_path
    return None


def compute_gripper_path(pose, grasp, pos_step_size=0.02):
    grasp_pose = pbu.multiply(pose.get_pose(), pbu.invert(grasp.grasp))
    pregrasp_pose = pbu.multiply(pose.get_pose(), pbu.invert(grasp.pregrasp))
    gripper_path = list(
        pbu.interpolate_poses(grasp_pose, pregrasp_pose, pos_step_size=pos_step_size)
    )
    return gripper_path


def workspace_collision(
    sim: SimulatorInstance,
    gripper_path,
    grasp=None,
    open_gripper=True,
    obstacles=[],
    max_distance=0.0,
):
    gripper = sim.get_component(sim.gripper_group)

    if open_gripper:
        _, open_conf = sim.get_group_limits(sim.gripper_group)
        gripper_joints = sim.get_component_joints(sim.gripper_group)
        pbu.set_joint_positions(gripper, gripper_joints, open_conf, client=sim.client)

    tool_parent = pbu.get_link_parent(
        sim.robot, sim.get_group_joints(sim.arm_group)[0], client=sim.client
    )
    parent_from_tool = pbu.get_link_pose(sim.robot, tool_parent, client=sim.client)

    parts = [gripper]
    if grasp is not None:
        parts.append(grasp.body)

    for i, gripper_pose in enumerate(gripper_path):
        pbu.set_pose(
            gripper,
            pbu.multiply(gripper_pose, pbu.invert(parent_from_tool)),
            client=sim.client,
        )
        if grasp is not None:
            pbu.set_pose(
                grasp.body, pbu.multiply(gripper_pose, grasp.value), client=sim.client
            )

        distance = (
            (COLLISION_DISTANCE + COLLISION_EPSILON)
            if (i == len(gripper_path) - 1)
            else max_distance
        )
        if any(
            pbu.pairwise_collisions(
                part, obstacles, max_distance=distance, client=sim.client
            )
            for part in parts
        ):
            return True
    return False


def plan_prehensile(
    sim: SimulatorInstance, obj, pose, grasp, environment=[], debug=False, **kwargs
):

    pbu.set_pose(obj, pose, client=sim.client)
    gripper_path = compute_gripper_path(pose, grasp)
    gripper_waypoints = gripper_path[:1] + gripper_path[-1:]
    if workspace_collision(
        sim, gripper_path, grasp=None, obstacles=environment, **kwargs
    ):
        return None

    arm_path = plan_workspace_motion(
        sim.robot,
        gripper_waypoints,
        attachment=None,
        obstacles=environment,
        debug=debug,
    )
    return arm_path


def get_plan_pick_fn(sim: SimulatorInstance, environment: List[int] = [], debug=False):
    environment = environment

    def fn(obj: int, pose: pbu.Pose, grasp: Grasp):
        arm_path = plan_prehensile(
            sim, obj, pose, grasp, environment=environment, debug=debug
        )

        if arm_path is None:
            return None

        arm_traj = Trajectory(
            sim.robot,
            sim.get_group_joints(sim.arm_group),
            arm_path[::-1],
            context=[pose],
            velocity_scale=0.25,
        )
        arm_conf = arm_traj[0]
        switch = ActivateGrasp(sim.robot, sim.tool_link, obj)
        commands = [arm_traj, switch, arm_traj.reverse()]
        sequence = Sequence(commands=commands, name="pick-{}".format(obj))
        return (arm_conf, sequence)

    return fn


#######################################################


def get_plan_place_fn(sim: SimulatorInstance, environment=[], debug=False):
    environment = environment

    def fn(obj: int, pose: pbu.Pose, grasp: Grasp):
        arm_path = plan_prehensile(
            sim.robot, obj, pose, grasp, environment=environment, debug=debug
        )
        if arm_path is None:
            return None

        arm_traj = Trajectory(
            sim.robot,
            sim.get_group_joints(sim.arm_group),
            arm_path,
            velocity_scale=0.25,
        )
        arm_conf = arm_traj.first()
        switch = DeactivateGrasp(sim.robot, sim.tool_link, obj)
        commands = [arm_traj, switch, arm_traj.reverse()]
        sequence = Sequence(commands=commands, name="place-{}".format(obj))

        return (arm_conf, sequence)

    return fn


def get_random_placement_pose(obj, surface_aabb, client):
    return pbu.Pose(
        point=pbu.Point(
            x=random.uniform(surface_aabb.lower[0], surface_aabb.upper[0]),
            y=random.uniform(surface_aabb.upper[1], surface_aabb.upper[1]),
            z=pbu.stable_z_on_aabb(obj, surface_aabb, client=client),
        ),
        euler=pbu.Euler(yaw=random.uniform(0, math.pi * 2)),
    )


def get_pick_place_plan(
    sim: SimulatorInstance,
    belief: WorldBelief,
    obj: int,
    grasp_sampler: Callable[[int, pbu.AABB, pbu.Pose], Grasp],
    motion_planner: Callable[[Conf, Conf, List[Attachment]], Trajectory],
    max_grasp_attempts=5,
    max_pick_attempts=5,
    max_place_attempts=5,
) -> Sequence:

    # Only plan with the digital twin
    assert not sim.real_robot

    # Make sure the simulator matches the belief
    sim.set_belief(belief)

    body_saver = pbu.WorldSaver(client=sim.client)

    obstacles = sim.movable_objects + [sim.table]

    pick_planner = get_plan_pick_fn(sim, environment=obstacles)
    place_planner = get_plan_place_fn(sim, environment=obstacles)

    q1 = Conf(
        sim.robot,
        sim.get_group_joints(sim.arm_group),
        sim.get_group_positions(sim.arm_group),
    )

    for gi in range(max_grasp_attempts):
        print("[Planner] grasp attempt " + str(gi))
        body_saver.restore()
        grasp: Grasp = grasp_sampler(obj)

        print("[Planner] finding pick plan for grasp " + str(grasp))
        for _ in range(max_pick_attempts):
            body_saver.restore()
            pick = pick_planner(obj, grasp)
            if pick is not None:
                break

        if pick is None:
            continue

        q2, at1 = pick
        q2.assign(sim)

        for _ in range(max_place_attempts):
            placement_pose = get_random_placement_pose(
                obj, pbu.get_aabb(sim.table), sim.client
            )
            body_saver.restore()
            place = place_planner(obj, placement_pose, grasp)

            if place is not None:
                break

        if place is None:
            continue

        q3, at2 = place
        q3.assign(sim)

        attachment = grasp.create_attachment(sim.robot, link=sim.tool_link)

        print("[Planner] finding pick motion plan")
        body_saver.restore()
        motion_plan1 = motion_planner(q1, q2)
        if motion_plan1 is None:
            continue

        print("[Planner] finding place motion plan")
        body_saver.restore()
        motion_plan2 = motion_planner(q2, q3, attachments=[attachment])
        if motion_plan2 is None:
            continue

        return Sequence([motion_plan1, at1, motion_plan2, at2], name="pick-place")
    return None
