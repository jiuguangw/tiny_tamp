from __future__ import annotations

import math

import numpy as np
from structs import (
    ARM_GROUP,
    COLLISION_DISTANCE,
    COLLISION_EPSILON,
    GRIPPER_GROUP,
    MAX_IK_DISTANCE,
    MAX_IK_TIME,
    SELF_COLLISIONS,
    Grasp,
    GroupConf,
    GroupTrajectory,
    ParentBody,
    RelativePose,
    Sequence,
    SimulatorInstance,
    Switch,
)

import tiny_tamp.pb_utils as pbu
from tiny_tamp.inverse_kinematics.franka_panda.ik import PANDA_INFO
from tiny_tamp.inverse_kinematics.ikfast import (
    closest_inverse_kinematics,
    get_ik_joints,
    ikfast_inverse_kinematics,
)


def get_plan_motion_fn(sim: SimulatorInstance, environment=[], debug=False, **kwargs):

    def fn(q1, q2, attachments=[]):
        print("Plan motion fn {}->{}".format(q1, q2))

        obstacles = list(environment)
        attached = {attachment.child for attachment in attachments}
        obstacles = set(obstacles) - attached
        q1.assign(**kwargs)

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
            **kwargs,
        )

        if path is None:
            for conf in [q1, q2]:
                conf.assign(**kwargs)
                for attachment in attachments:
                    attachment.assign(**kwargs)
            return None

        sequence = Sequence(
            commands=[
                GroupTrajectory(sim, ARM_GROUP, path, **kwargs),
            ],
            name="move-{}".format(ARM_GROUP),
        )
        return sequence

    return fn


def plan_workspace_motion(
    sim: SimulatorInstance,
    tool_waypoints,
    attachment=None,
    obstacles=[],
    max_attempts=2,
    debug=False,
    **kwargs,
):

    assert tool_waypoints

    tool_link = sim.tool_link
    ik_joints = get_ik_joints(sim.robot, PANDA_INFO, tool_link, **kwargs)  # Arm + torso
    fixed_joints = set(ik_joints) - set(sim.robot.get_group_joints(ARM_GROUP, **kwargs))
    arm_joints = [j for j in ik_joints if j not in fixed_joints]  # Arm only
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
        **kwargs,
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
            **kwargs,
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
                        **kwargs,
                    ),
                    None,
                )
                if arm_conf is None:
                    break

                arm_conf = extract_arm_conf(arm_conf)
                if collision_fn(arm_conf):
                    if debug:
                        pbu.wait_if_gui(**kwargs)
                    continue
                arm_waypoints.append(arm_conf)
            else:
                pbu.set_joint_positions(
                    sim.robot, arm_joints, arm_waypoints[-1], **kwargs
                )
                if attachment is not None:
                    attachment.assign(**kwargs)
                if any(
                    pbu.pairwise_collisions(
                        part,
                        obstacles,
                        max_distance=(COLLISION_DISTANCE + COLLISION_EPSILON),
                        **kwargs,
                    )
                    for part in parts
                ):
                    if debug:
                        pbu.wait_if_gui(**kwargs)
                    continue
                arm_path = pbu.interpolate_joint_waypoints(
                    sim.robot, arm_joints, arm_waypoints, **kwargs
                )

                if any(collision_fn(q) for q in arm_path):
                    if debug:
                        pbu.wait_if_gui(**kwargs)
                    continue

                print(
                    "Found path with {} waypoints and {} configurations after {} attempts".format(
                        len(arm_waypoints), len(arm_path), attempts + 1
                    )
                )

                return arm_path
    print("[plan_workspace_motion] max_attempts reached")
    return None


def compute_gripper_path(pose, grasp, pos_step_size=0.02):
    grasp_pose = pbu.multiply(pose.get_pose(), pbu.invert(grasp.grasp))
    pregrasp_pose = pbu.multiply(pose.get_pose(), pbu.invert(grasp.pregrasp))
    gripper_path = list(
        pbu.interpolate_poses(grasp_pose, pregrasp_pose, pos_step_size=pos_step_size)
    )
    return gripper_path


def workspace_collision(
    robot,
    gripper_path,
    grasp=None,
    open_gripper=True,
    obstacles=[],
    max_distance=0.0,
    **kwargs,
):
    gripper = robot.get_component(GRIPPER_GROUP, **kwargs)

    if open_gripper:
        _, open_conf = robot.get_group_limits(GRIPPER_GROUP, **kwargs)
        gripper_joints = robot.get_component_joints(GRIPPER_GROUP, **kwargs)
        pbu.set_joint_positions(gripper, gripper_joints, open_conf, **kwargs)

    parent_from_tool = robot.get_parent_from_tool(**kwargs)
    parts = [gripper]
    if grasp is not None:
        parts.append(grasp.body)
    for i, gripper_pose in enumerate(gripper_path):
        pbu.set_pose(
            gripper, pbu.multiply(gripper_pose, pbu.invert(parent_from_tool)), **kwargs
        )
        if grasp is not None:
            pbu.set_pose(grasp.body, pbu.multiply(gripper_pose, grasp.value), **kwargs)

        distance = (
            (COLLISION_DISTANCE + COLLISION_EPSILON)
            if (i == len(gripper_path) - 1)
            else max_distance
        )
        if any(
            pbu.pairwise_collisions(part, obstacles, max_distance=distance, **kwargs)
            for part in parts
        ):
            return True
    return False


def create_grasp_attachment(sim: SimulatorInstance, grasp: Grasp, **kwargs):
    return grasp.create_attachment(sim.robot, link=sim.tool_link)


def plan_prehensile(sim, obj, pose, grasp, environment=[], debug=False, **kwargs):

    pose.assign(**kwargs)

    gripper_path = compute_gripper_path(pose, grasp)
    gripper_waypoints = gripper_path[:1] + gripper_path[-1:]
    if workspace_collision(
        sim, gripper_path, grasp=None, obstacles=environment, **kwargs
    ):
        if debug:
            print("[plan_prehensile] workspace collision")
        pbu.wait_if_gui(**kwargs)
        return None

    create_grasp_attachment(sim.robot, grasp)
    arm_path = plan_workspace_motion(
        sim.robot,
        gripper_waypoints,
        attachment=None,
        obstacles=environment,
        debug=debug,
    )
    return arm_path


def get_plan_pick_fn(sim, environment=[], debug=False, **kwargs):
    environment = environment

    def fn(obj, pose, grasp):
        arm_path = plan_prehensile(
            sim, obj, pose, grasp, environment=environment, debug=debug, **kwargs
        )

        if arm_path is None:
            return None

        arm_traj = GroupTrajectory(
            sim,
            ARM_GROUP,
            arm_path[::-1],
            context=[pose],
            velocity_scale=0.25,
            **kwargs,
        )
        arm_conf = arm_traj.first()

        closed_conf = grasp.closed_position * np.ones(
            len(sim.get_group_joints(GRIPPER_GROUP, **kwargs))
        )
        gripper_traj = GroupTrajectory(
            sim,
            GRIPPER_GROUP,
            path=[closed_conf],
            contexts=[pose],
            contact_links=sim.get_finger_links(
                sim.get_group_joints(GRIPPER_GROUP, **kwargs), **kwargs
            ),
            time_after_contact=1e-1,
            **kwargs,
        )
        switch = Switch(
            obj,
            parent=ParentBody(
                body=sim.robot,
                link=sim.tool_link,
                **kwargs,
            ),
        )

        commands = [arm_traj, switch, gripper_traj, arm_traj.reverse(**kwargs)]
        sequence = Sequence(commands=commands, name="pick-{}".format(obj))
        return (arm_conf, sequence)

    return fn


#######################################################


def get_plan_place_fn(robot, environment=[], debug=False, **kwargs):
    environment = environment

    def fn(obj, pose, grasp):
        arm_path = plan_prehensile(
            robot, obj, pose, grasp, environment=environment, debug=debug, **kwargs
        )
        if arm_path is None:
            print("[plan_place_fn] arm_path is None")
            return None

        arm_traj = GroupTrajectory(
            robot,
            ARM_GROUP,
            arm_path[::-1],
            context=[grasp],
            velocity_scale=0.25,
            **kwargs,
        )
        arm_conf = arm_traj.first()

        _, open_conf = robot.get_group_limits(GRIPPER_GROUP, **kwargs)
        gripper_traj = GroupTrajectory(
            robot,
            GRIPPER_GROUP,
            path=[open_conf],
            contexts=[grasp],
            **kwargs,
        )
        switch = Switch(obj, parent=None)

        commands = [arm_traj, gripper_traj, switch, arm_traj.reverse(**kwargs)]
        sequence = Sequence(commands=commands, name="place-{}".format(obj))

        return (arm_conf, sequence)

    return fn


def get_pick_place_plan(
    sim: SimulatorInstance,
    obj: int,
    placement_pose: pbu.Pose,
    grasp_sampler,
    motion_planner,
    csp_debug=False,
):

    MAX_GRASP_ATTEMPTS = 100
    MAX_PICK_ATTEMPTS = 1
    MAX_PLACE_ATTEMPTS = 1

    body_saver = pbu.WorldSaver(client=sim.client)
    obj_aabb, obj_pose = pbu.get_aabb(obj, client=sim.client), pbu.get_pose(
        obj, client=sim.client
    )

    obstacles = sim.movable_objects + [sim.table]

    pick_planner = get_plan_pick_fn(
        sim,
        environment=obstacles,
        max_attempts=MAX_PICK_ATTEMPTS,
        debug=csp_debug,
        client=sim.client,
    )
    place_planner = get_plan_place_fn(
        sim,
        environment=obstacles,
        max_attempts=MAX_PLACE_ATTEMPTS,
        debug=csp_debug,
        client=sim.client,
    )

    pose = RelativePose(obj, client=sim.client)
    q1 = GroupConf(
        sim.robot, ARM_GROUP, sim.robot.arm_group.get_joint_values(), client=sim.client
    )

    for gi in range(MAX_GRASP_ATTEMPTS):
        print("[Planner] grasp attempt " + str(gi))
        body_saver.restore()
        (grasp,) = next(grasp_sampler(obj, obj_aabb, obj_pose))

        print("[Planner] finding pick plan for grasp " + str(grasp))
        for _ in range(MAX_PICK_ATTEMPTS):
            body_saver.restore()
            pick = pick_planner(obj, pose, grasp)
            if pick is not None:
                break

        if pick is None:
            continue

        q2, at1 = pick
        q2.assign(client=sim.client)

        print("[Planner] finding place plan")
        for _ in range(MAX_PLACE_ATTEMPTS):
            place_rp = RelativePose(
                obj, parent=None, relative_pose=placement_pose, client=sim.client
            )
            print("[Place Planner] Placement for pose: " + str(place_rp))
            body_saver.restore()
            place = place_planner(obj, place_rp, grasp)

            if place is not None:
                break

        if place is None:
            continue

        q3, at2 = place
        q3.assign(client=sim.client)

        if csp_debug:
            pbu.wait_if_gui("Placing like this", client=sim.client)

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

        return Sequence([motion_plan1, at1, motion_plan2, at2])
    return None
