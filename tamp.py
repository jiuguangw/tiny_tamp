from __future__ import annotations

import argparse
import numpy as np
import copy
import pb_utils as pbu
from planning import get_pick_place_plan, get_plan_motion_fn
import json
import sys
from structs import DEFAULT_JOINT_POSITIONS, DEFAULT_TS, GRIPPER_GROUP, Grasp, ObjectState, SimulatorInstance, SimulatorState, WorldBelief
from typing import Callable

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
    object_states = [
        ObjectState(),
        ObjectState()
    ]

    return WorldBelief(
        object_states=object_states,
        robot_state=DEFAULT_JOINT_POSITIONS
    )

def dummy_get_goal(belief: WorldBelief) -> WorldBelief:
    new_belief = copy.deepcopy(belief)
    # The goal is to shift the red and blue blocks a little
    new_belief.object_states[0].pose[0][0] += 0.1
    new_belief.object_states[0].pose[0][0] -= 0.1
    return new_belief

def get_grasp_gen_fn(sim: SimulatorInstance, belief: WorldBelief) -> Callable[[int], Grasp]:

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
    goal_belief = dummy_get_goal(belief) # Get your rearrangement goal

    # Typically maintain two instances. One for visualizing the plan before execution and the other for planning
    assert not (args.vis_belief and args.vis_sim) # You can only vis one thing in pb
    sim_instance = SimulatorInstance().from_belief(belief, gui = args.vis_belief)
    twin_sim_instance = SimulatorInstance().from_belief(belief, gui = args.vis_sim)

    motion_planner = get_plan_motion_fn(
        twin_sim_instance, 
    )

    grasp_sampler = get_grasp_gen_fn(
        twin_sim_instance, belief
    )

    plan_components = []
    for object_name in belief.object_states:
        if(pbu.get_pose_distance(belief.object_states[object_name], goal_belief.object_states[object_name]) > 1e-3):
            subplan, statistics = get_pick_place_plan(motion_planner, grasp_sampler, )
            if(subplan is None):
                print("Planning failure")
                print(json.dumps(statistics))
                sys.exit()

            plan_components.append(subplan)

    sim_state = SimulatorState(sim_instance)
    for sequence in plan_components:
        for i, _ in enumerate(sequence.iterate(sim_state, real_controller=args.real_robot)):
            sim_state.propagate()
            pbu.wait_for_duration(DEFAULT_TS)
    

if __name__ == "__main__":
    main()
