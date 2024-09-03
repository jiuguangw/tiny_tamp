from __future__ import annotations

import copy
import itertools
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
from tiny_tamp.motion_planning.motion_planners.rrt_connect import birrt
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

        path = plan_joint_motion(
            sim,
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
        )

        if path is None:
            return None

        return Trajectory(sim.robot, q2.joints, path, attachments=attachments)

    return fn


def plan_workspace_motion(
    sim: SimulatorInstance,
    tool_waypoints: List[pbu.Pose],
    attachment: Attachment = None,
    obstacles: List[int] = [],
    max_attempts=2,
    debug=False,
) -> List[List[float]]:
    """Return a joint path that moves the tool along the tool waypoints.

    This is useful if you want to move the gripper in a straight line
    path.
    """
    print("Tool waypoints")
    print(len(tool_waypoints))
    assert tool_waypoints

    tool_link = sim.tool_link
    ik_joints = get_ik_joints(sim.robot, PANDA_INFO, tool_link, client=sim.client)
    fixed_joints = set(ik_joints) - set(sim.get_group_joints(sim.arm_group))
    arm_joints = [j for j in ik_joints if j not in fixed_joints]
    extract_arm_conf = lambda q: [
        p for j, p in pbu.safe_zip(ik_joints, q) if j not in fixed_joints
    ]

    parts = [sim.robot] + ([] if attachment is None else [attachment.child])
    collision_fn = get_collision_fn(
        sim,
        arm_joints,
        obstacles=obstacles,
        attachments=[],
        self_collisions=SELF_COLLISIONS,
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


def compute_gripper_path(pose: pbu.Pose, grasp: Grasp, pos_step_size=0.02):
    grasp_pose = pbu.multiply(pose, pbu.invert(grasp.attachment.parent_T_child))
    pregrasp_pose = pbu.multiply(
        pose, pbu.invert(grasp.get_pregrasp_pose(grasp.attachment.parent_T_child))
    )
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

    parent_link = sim.get_group_parent(sim.gripper_group)
    parent_from_tool = pbu.get_relative_pose(
        sim.robot, sim.tool_link, parent_link, client=sim.client
    )

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

            print("[workspace_collision] gripper path in collision")
            return True

    return False


def plan_prehensile(
    sim: SimulatorInstance,
    obj: int,
    pose: pbu.Pose,
    grasp: Grasp,
    environment: List[int] = [],
    debug: bool = False,
):

    pbu.set_pose(obj, pose, client=sim.client)
    gripper_path = compute_gripper_path(pose, grasp)
    if workspace_collision(sim, gripper_path, grasp=None, obstacles=environment):
        print("[plan_prehensile] gripper path in collision")
        return None

    arm_path = plan_workspace_motion(
        sim,
        gripper_path,
        attachment=None,
        obstacles=environment,
        debug=debug,
    )

    if arm_path is None:
        print("[plan_prehensile] arm path none")

    return arm_path


def get_plan_pick_fn(sim: SimulatorInstance, environment: List[int] = [], debug=False):
    environment = environment

    def fn(obj: int, pose: pbu.Pose, grasp: Grasp):
        obstacles = list(set(environment) - {obj})
        arm_path = plan_prehensile(
            sim, obj, pose, grasp, environment=obstacles, debug=debug
        )

        if arm_path is None:
            print("[get_plan_pick_fn] failed to find a plan")
            return None

        arm_traj = Trajectory(
            sim.robot,
            sim.get_group_joints(sim.arm_group),
            arm_path[::-1],
        )
        arm_conf = Conf(
            sim.robot, sim.get_group_joints(sim.arm_group), arm_traj.path[0]
        )
        switch = ActivateGrasp(sim.robot, sim.tool_link, obj)
        reverse_traj = copy.deepcopy(arm_traj).reverse()
        reverse_traj.attachments = [grasp.attachment]
        commands = [arm_traj, switch, reverse_traj]
        sequence = Sequence(commands=commands, name="pick-{}".format(obj))
        return (arm_conf, sequence)

    return fn


#######################################################


def get_plan_place_fn(sim: SimulatorInstance, environment=[], debug=False):
    environment = environment

    def fn(obj: int, pose: pbu.Pose, grasp: Grasp):

        obstacles = list(set(environment) - {obj})
        arm_path = plan_prehensile(
            sim, obj, pose, grasp, environment=obstacles, debug=debug
        )
        if arm_path is None:
            return None

        arm_traj = Trajectory(
            sim.robot,
            sim.get_group_joints(sim.arm_group),
            arm_path[::-1],
            attachments=[grasp.attachment],
        )
        arm_conf = Conf(
            sim.robot, sim.get_group_joints(sim.arm_group), arm_traj.path[0]
        )
        switch = DeactivateGrasp(sim.robot, sim.tool_link, obj)
        reverse_traj = copy.deepcopy(arm_traj).reverse()
        reverse_traj.attachments = []
        commands = [arm_traj, switch, reverse_traj]
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
    placement_location=None,
) -> Sequence:

    # Only plan with the digital twin
    assert not sim.real_robot

    # Make sure the simulator matches the belief
    sim.set_belief(belief)

    body_saver = pbu.WorldSaver(client=sim.client)

    obstacles = sim.movable_objects + [sim.table]
    obj_pose = pbu.get_pose(obj, client=sim.client)

    pick_planner = get_plan_pick_fn(sim, environment=obstacles)
    place_planner = get_plan_place_fn(sim, environment=obstacles)

    statistics = {}
    q1 = Conf(
        sim.robot,
        sim.get_group_joints(sim.arm_group),
        sim.get_group_positions(sim.arm_group),
    )

    for gi in range(max_grasp_attempts):
        print("[Planner] grasp attempt " + str(gi))
        body_saver.restore()
        grasp = grasp_sampler(obj)

        print("[Planner] finding pick plan for grasp " + str(grasp))
        for _ in range(max_pick_attempts):
            body_saver.restore()
            pick = pick_planner(obj, obj_pose, grasp)
            if pick is not None:
                break

        if pick is None:
            continue

        q2, at1 = pick
        q2.assign(sim)

        for _ in range(max_place_attempts):
            if placement_location is not None:
                placement_pose = placement_location
            else:
                placement_pose = get_random_placement_pose(
                    obj, pbu.get_aabb(sim.table, client=sim.client), sim.client
                )
            body_saver.restore()
            place = place_planner(obj, placement_pose, grasp)

            if place is not None:
                break

        if place is None:
            continue

        q3, at2 = place
        q3.assign(sim)

        print("[Planner] finding pick motion plan")
        body_saver.restore()
        motion_plan1 = motion_planner(q1, q2)
        if motion_plan1 is None:
            continue

        print("[Planner] finding place motion plan")
        body_saver.restore()
        motion_plan2 = motion_planner(q2, q3, attachments=[grasp.attachment])
        if motion_plan2 is None:
            continue

        return (
            Sequence([motion_plan1, at1, motion_plan2, at2], name=f"pick-place({obj})"),
            statistics,
        )
    return None, statistics


#################Motion Planner######################


def get_collision_fn(
    sim: SimulatorInstance,
    joints: List[int],
    obstacles: List[int] = [],
    attachments: List[Attachment] = [],
    self_collisions: bool = True,
    disabled_collisions=set(),
    custom_limits={},
    use_aabb=False,
    cache=False,
    max_distance=pbu.MAX_DISTANCE,
    extra_collisions=None,
):
    check_link_pairs = (
        pbu.get_self_link_pairs(
            sim.robot, joints, disabled_collisions, client=sim.client
        )
        if self_collisions
        else []
    )
    moving_links = frozenset(
        link
        for link in pbu.get_moving_links(sim.robot, joints, client=sim.client)
        if pbu.can_collide(sim.robot, link, client=sim.client)
    )
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [pbu.CollisionPair(sim.robot, moving_links)] + list(
        map(pbu.parse_body, attached_bodies)
    )

    get_obstacle_aabb = pbu.cached_fn(
        pbu.get_buffered_aabb,
        cache=cache,
        max_distance=max_distance / 2.0,
        client=sim.client,
    )
    limits_fn = pbu.get_limits_fn(
        sim.robot, joints, custom_limits=custom_limits, client=sim.client
    )

    def collision_fn(q, verbose=False):

        if limits_fn(q):
            return True

        pbu.set_joint_positions(sim.robot, joints, q, client=sim.client)

        for attachment in attachments:
            world_T_child = pbu.multiply(
                pbu.get_link_pose(
                    attachment.parent, attachment.parent_link, client=sim.client
                ),
                attachment.parent_T_child,
            )
            pbu.set_pose(attachment.child, world_T_child, client=sim.client)

            # Check if the attachment is in collision with objects in the environment
            if any(
                pbu.pairwise_collision(attachment.child, body, client=sim.client)
                for body in obstacles
            ):
                return True

        if extra_collisions is not None and extra_collisions(client=sim.client):
            return True

        get_moving_aabb = pbu.cached_fn(
            pbu.get_buffered_aabb,
            cache=True,
            max_distance=max_distance / 2.0,
            client=sim.client,
        )

        for link1, link2 in check_link_pairs:
            if (
                not use_aabb
                or pbu.aabb_overlap(
                    get_moving_aabb(sim.robot), get_moving_aabb(sim.robot)
                )
            ) and pbu.pairwise_link_collision(
                sim.robot, link1, sim.robot, link2, client=sim.client
            ):
                print("Link on link collision")
                print(sim.robot, link1, sim.robot, link2)
                return True

        for body1, body2 in itertools.product(moving_bodies, obstacles):
            if (
                not use_aabb
                or pbu.aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))
            ) and pbu.pairwise_collision(body1, body2, client=sim.client):
                print("Body on body collision")
                print(body1, body2)
                return True
        return False

    return collision_fn


def check_initial_end(
    sim: SimulatorInstance,
    joints: List[int],
    start_conf: List[float],
    end_conf: List[float],
    collision_fn: Callable[[List[float]], bool],
    debug=False,
    verbose=True,
):
    if collision_fn(start_conf, verbose=verbose):
        print("Warning: initial configuration is in collision {}".format(start_conf))
        if debug:
            pbu.set_joint_positions(sim.robot, joints, start_conf, client=sim.client)
            pbu.wait_if_gui(client=sim.client)
        return False
    if collision_fn(end_conf, verbose=verbose):
        print("Warning: end configuration is in collision {}".format(end_conf))
        if debug:
            pbu.set_joint_positions(sim.robot, joints, end_conf, client=sim.client)
            pbu.wait_if_gui(client=sim.client)
        return False
    return True


def plan_joint_motion(
    sim: SimulatorInstance,
    joints: List[int],
    end_conf: List[float],
    obstacles: List[int] = [],
    attachments: List[Attachment] = [],
    self_collisions: bool = True,
    disabled_collisions=set(),
    weights=None,
    resolutions=None,
    max_distance=pbu.MAX_DISTANCE,
    use_aabb=False,
    cache=True,
    custom_limits={},
    extra_collisions=None,
    debug=False,
    **kwargs,
):
    assert len(joints) == len(end_conf)
    if (weights is None) and (resolutions is not None):
        weights = np.reciprocal(resolutions)
    sample_fn = pbu.get_sample_fn(
        sim.robot, joints, custom_limits=custom_limits, client=sim.client
    )
    distance_fn = pbu.get_distance_fn(
        sim.robot, joints, weights=weights, client=sim.client
    )
    extend_fn = pbu.get_extend_fn(
        sim.robot, joints, resolutions=resolutions, client=sim.client
    )
    collision_fn = get_collision_fn(
        sim,
        joints,
        obstacles,
        attachments,
        self_collisions,
        disabled_collisions,
        custom_limits=custom_limits,
        max_distance=max_distance,
        use_aabb=use_aabb,
        cache=cache,
        extra_collisions=extra_collisions,
    )

    start_conf = pbu.get_joint_positions(sim.robot, joints, client=sim.client)
    if not check_initial_end(
        sim, joints, start_conf, end_conf, collision_fn, debug=debug
    ):
        return None

    return birrt(
        start_conf,
        end_conf,
        distance_fn,
        sample_fn,
        extend_fn,
        collision_fn,
        client=sim.client,
        **kwargs,
    )
