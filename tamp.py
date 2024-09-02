from __future__ import annotations

import argparse
import copy
import json
import sys
from typing import Callable

import numpy as np

import tiny_tamp.pb_utils as pbu
from tiny_tamp.planning import get_pick_place_plan, get_plan_motion_fn
from tiny_tamp.structs import (
    DEFAULT_JOINT_POSITIONS,
    DEFAULT_TS,
    GRIPPER_GROUP,
    TABLE_POSE,
    GoalBelief,
    Grasp,
    ObjectState,
    SimulatorInstance,
    SimulatorState,
    WorldBelief,
)


def create_args():
    """Creates the arguments for the experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-robot", action="store_true", help="View the pybullet gui when planning"
    )

    parser.add_argument(
        "--vis-sim", action="store_true", help="View the pybullet gui when planning"
    )

    parser.add_argument(
        "--vis-belief", action="store_true", help="View the pybullet gui when planning"
    )

    args = parser.parse_args()
    return args


def dummy_perception() -> WorldBelief:

    box_size = 0.05

    object1 = ObjectState(
        create_object=lambda client: pbu.create_box(
            w=box_size, l=box_size, h=box_size, color=pbu.RED, client=client
        ),
        pose=pbu.multiply(pbu.Pose(pbu.Point(y=0.1, z=box_size)), TABLE_POSE),
        category="red_box",
    )

    object2 = ObjectState(
        create_object=lambda client: pbu.create_box(
            w=box_size, l=box_size, h=box_size, color=pbu.BLUE, client=client
        ),
        pose=pbu.multiply(pbu.Pose(pbu.Point(y=-0.1, z=box_size)), TABLE_POSE),
        category="blue_box",
    )

    object_states = [object1, object2]

    return WorldBelief(object_states=object_states, robot_state=DEFAULT_JOINT_POSITIONS)


def dummy_get_goal(belief: WorldBelief) -> GoalBelief:
    new_belief = copy.deepcopy(belief)

    # The goal is to shift the red and blue blocks a little
    new_belief.object_states[0].pose = pbu.multiply(
        pbu.Pose(pbu.Point(x=0.1)), new_belief.object_states[0].pose
    )
    new_belief.object_states[1].pose = pbu.multiply(
        pbu.Pose(pbu.Point(x=-0.1)), new_belief.object_states[1].pose
    )
    return new_belief


def get_grasp_gen_fn(
    sim: SimulatorInstance, belief: WorldBelief
) -> Callable[[int], Grasp]:

    def gen_fn(obj: int) -> Grasp:
        closed_conf, _ = sim.get_group_limits(GRIPPER_GROUP)
        closed_position = closed_conf[0] * (1 + 5e-2)
        grasp_pose = pbu.multiply(
            pbu.Pose(euler=pbu.Euler(pitch=-np.pi / 2.0)),
            pbu.Pose(pbu.Point(z=-0.01)),
        )
        return Grasp(closed_position, grasp_pose)

    return gen_fn


def main():
    args = create_args()

    # This is where you put your perception. If you wan to do grasp sampling here, you can just add it to the object states
    belief = dummy_perception()
    goal_belief = dummy_get_goal(belief)  # Get your rearrangement goal

    # Typically maintain two instances. One for visualizing the plan before execution and the other for planning
    assert not (args.vis_belief and args.vis_sim)  # You can only vis one thing in pb
    sim_instance = SimulatorInstance.from_belief(belief, gui=args.vis_sim)
    twin_sim_instance = SimulatorInstance.from_belief(belief, gui=args.vis_belief)

    pbu.wait_if_gui("Press enter to start planning", client=sim_instance.client)

    motion_planner = get_plan_motion_fn(
        twin_sim_instance,
    )

    grasp_sampler = get_grasp_gen_fn(twin_sim_instance, belief)

    plan_components = []
    for goal_object_state in goal_belief.object_states:

        # Assuming all objects have unique categories
        belief_object_index, belief_object = [
            (obj_idx, obj)
            for obj_idx, obj in enumerate(belief.object_states)
            if obj.category == goal_object_state.category
        ][0]

        pos_error, ori_error = pbu.get_pose_distance(
            belief_object.pose,
            goal_object_state.pose,
        )
        if (pos_error > 1e-2) or (ori_error > 1e-2):
            subplan, statistics = get_pick_place_plan(
                sim_instance,
                sim_instance.movable_objects[belief_object_index],
                motion_planner,
                grasp_sampler,
            )
            if subplan is None:
                print("Planning failure")
                print(json.dumps(statistics))
                sys.exit()

            plan_components.append(subplan)

    sim_state = SimulatorState(sim_instance)

    for sequence in plan_components:
        for i, _ in enumerate(
            sequence.iterate(sim_state, real_controller=args.real_robot)
        ):
            sim_state.propagate()
            pbu.wait_for_duration(DEFAULT_TS)


if __name__ == "__main__":
    main()
