from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc

import tiny_tamp.pb_utils as pbu
from tiny_tamp.hardware.panda_sender import PandaSender

ARM_GROUP = "main_arm"
GRIPPER_GROUP = "main_gripper"
PANDA_TOOL_TIP = "panda_tool_tip"
PANDA_GROUPS = {
    "base": [],
    "main_arm": ["panda_joint{}".format(i) for i in range(1, 8)],
    "main_gripper": ["panda_finger_joint1", "panda_finger_joint2"],
}
DEFAULT_JOINT_POSITIONS = [
    -0.0806406098426434,
    -1.6722951504174777,
    0.07069076842695393,
    -2.7449419709102822,
    0.08184716251979611,
    1.7516337599063168,
    0.7849295270972781,
]
SELF_COLLISIONS = True
PANDA_IGNORE_COLLISIONS = {(6, 9)}
MAX_IK_TIME = 0.01
MAX_IK_DISTANCE = np.inf
MAX_TOOL_DISTANCE = np.inf
COLLISION_EPSILON = 1e-3
COLLISION_DISTANCE = 5e-3

ROBOT_URDF = "./models/franka_panda/panda.urdf"
TABLE_AABB = pbu.AABB(
    lower=(-0.50 / 2.0, -1.0 / 2.0, -0.03 / 2.0),
    upper=(0.50 / 2.0, 1.0 / 2.0, 0.03 / 2.0),
)
TABLE_POSE = pbu.Pose((0.45, 0, -TABLE_AABB.upper[2]))
DEFAULT_TS = 5e-3
PREGRASP_DISTANCE = 0.05


@dataclass
class ObjectState:
    # Function that creates a new object given a pybullet client
    create_object: Callable[[int], int]

    pose: pbu.Pose  # Current pose of the object
    category: str = ""  # Optional class label or object name


@dataclass
class WorldBelief:
    object_states: List[
        ObjectState
    ]  # Mapping from id in pybullet client to object state
    robot_state: List[float]  # Current robot movable joint positions
    gripper_open: bool = True


@dataclass
class GoalBelief(WorldBelief):
    """A belief about the goal state of the world. This is used to specify the
    desired object locations The semantics of this goal is a set of existential
    quantifiers.

    For each object in the belief with a particular category, the goal
    is to believe that there exists an object with the same category at
    the specified pose.

    More complex goal semantics will require a more capable planner
    """

    pass


@dataclass
class SimulatorInstance:
    client: int  # The pybullet phyiscs client
    robot: int  # The robot id in the pybullet client
    table: int  # The table id in the pybullet client
    movable_objects: List[int]  # Mapping from id in pybullet client to object state
    components: Dict[str, int] = field(
        default_factory=dict
    )  # Mapping from group name to component id

    real_robot: int = False
    sender: PandaSender = None

    arm_group = "main_arm"
    gripper_group = "main_gripper"

    group_joints: Dict[str, List[int]] = None

    @staticmethod
    def from_belief(belief: WorldBelief, gui=False, real_robot=False):
        client = bc.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        client.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        robot_body = pbu.load_pybullet(ROBOT_URDF, fixed_base=True, client=client)

        client.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraYaw=90,
            cameraPitch=-15,
            cameraTargetPosition=[-0.5, 0, 0.3],
        )
        table = pbu.create_box(
            *pbu.get_aabb_extent(TABLE_AABB), color=pbu.TAN, client=client
        )
        pbu.set_pose(table, TABLE_POSE, client=client)

        # Add the objects to the scene
        movable_objects = []
        for obj_state in belief.object_states:
            obj = obj_state.create_object(client)
            pbu.set_pose(obj, obj_state.pose, client=client)
            movable_objects.append(obj)

        sender = None
        if real_robot:
            sender = PandaSender()

        instance = SimulatorInstance(
            client,
            robot_body,
            table,
            movable_objects,
            real_robot=real_robot,
            sender=sender,
            group_joints={
                "main_arm": pbu.joints_from_names(
                    robot_body, PANDA_GROUPS["main_arm"], client=client
                ),
                "main_gripper": pbu.joints_from_names(
                    robot_body, PANDA_GROUPS["main_gripper"], client=client
                ),
            },
        )

        # Move robot to joint positions
        if belief.gripper_open:
            instance.open_gripper()
        else:
            instance.close_gripper()

        instance.set_group_positions(instance.arm_group, belief.robot_state)

        return instance

    def set_belief(self, belief: WorldBelief):
        assert len(belief.object_states) == len(self.movable_objects)

        for obj_state, obj_id in zip(belief.object_states, self.movable_objects):
            pbu.set_pose(obj_id, obj_state.pose, client=self.client)
        self.set_group_positions(self.arm_group, belief.robot_state)
        if belief.gripper_open:
            self.open_gripper()
        else:
            self.close_gripper()

    def get_group_parent(self, group):
        return pbu.get_link_parent(
            self.robot, self.group_joints[group][0], client=self.client
        )

    def get_group_subtree(self, group):
        return pbu.get_link_subtree(
            self.robot, self.get_group_parent(group), client=self.client
        )

    def get_component_mapping(self, group):
        assert group in self.components
        component_joints = pbu.get_movable_joints(
            self.components[group], draw=False, client=self.client
        )
        body_joints = pbu.get_movable_joint_descendants(
            self.robot,
            self.get_group_parent(group),
            client=self.client,
        )
        return OrderedDict(pbu.safe_zip(body_joints, component_joints))

    def get_component_joints(self, group):
        mapping = self.get_component_mapping(group)
        return list(map(mapping.get, self.group_joints[group]))

    def get_component(self, group, visual=True):
        if group not in self.components:
            component = pbu.clone_body(
                self.robot,
                links=self.get_group_subtree(group),
                visual=False,
                collision=True,
                client=self.client,
            )
            if not visual:
                pbu.set_all_color(component, pbu.TRANSPARENT)
            self.components[group] = component
        return self.components[group]

    @property
    def tool_link(self):
        return pbu.link_from_name(self.robot, PANDA_TOOL_TIP, client=self.client)

    def get_group_limits(self, group):
        return pbu.get_custom_limits(
            self.robot, self.group_joints[group], client=self.client
        )

    def get_group_joints(self, group):
        return pbu.joints_from_names(
            self.robot, PANDA_GROUPS[group], client=self.client
        )

    def open_gripper(self, dt: float = 0.0):
        current_conf = self.get_group_positions(GRIPPER_GROUP)
        _, open_conf = self.get_group_limits(GRIPPER_GROUP)
        self.set_group_positions(GRIPPER_GROUP, open_conf)

        # Open from current conf to open conf
        for joints in pbu.interpolate_joint_waypoints(
            self.robot,
            self.group_joints[GRIPPER_GROUP],
            [current_conf, open_conf],
            client=self.client,
        ):
            self.set_group_positions(GRIPPER_GROUP, joints)
            time.sleep(dt)

        if self.real_robot:
            self.sender.open_gripper()

    def close_gripper(self, dt: float = 0.0):
        current_conf = self.get_group_positions(GRIPPER_GROUP)
        closed_conf, _ = self.get_group_limits(GRIPPER_GROUP)

        # Close from current conf to closed conf or until collision
        for joints in pbu.interpolate_joint_waypoints(
            self.robot,
            self.group_joints[GRIPPER_GROUP],
            [current_conf, closed_conf],
            client=self.client,
        ):
            self.set_group_positions(GRIPPER_GROUP, joints)
            time.sleep(dt)

            # Stop closing if collision
            if any(
                pbu.pairwise_collision(
                    body1=self.robot, body2=other, client=self.client
                )
                for other in self.movable_objects
            ):
                break

        if self.real_robot:
            self.sender.close_gripper()

    def get_group_positions(self, group: str) -> List[float]:
        return pbu.get_joint_positions(
            self.robot, self.group_joints[group], client=self.client
        )

    def set_group_positions(self, group: str, positions: List[float]):
        pbu.set_joint_positions(
            self.robot, self.group_joints[group], positions, client=self.client
        )

        if self.real_robot:
            self.sender.command_arm(positions)

    def command_trajectory(self, trajectory: Trajectory, dt: float = 0.0):
        named_positions = []
        for positions in trajectory.path:
            self.set_group_positions(ARM_GROUP, positions)
            named_positions.append(
                {
                    name: position
                    for name, position in zip(PANDA_GROUPS[ARM_GROUP], positions)
                }
            )
            for attachment in trajectory.attachments:
                pbu.set_pose(
                    attachment.child,
                    pbu.multiply(
                        pbu.get_link_pose(
                            attachment.parent,
                            attachment.parent_link,
                            client=self.client,
                        ),
                        attachment.parent_T_child,
                    ),
                    client=self.client,
                )
            time.sleep(dt)
            # pbu.wait_if_gui("Press enter to continue", client=self.client)

        if self.real_robot:
            self.sender.execute_position_path(named_positions)

    def execute_command(self, command, dt=0.05):
        print("Executing command: {}".format(command))
        if isinstance(command, Trajectory):
            self.command_trajectory(command, dt=dt)
        elif isinstance(command, ActivateGrasp):
            self.close_gripper(dt=dt)
        elif isinstance(command, DeactivateGrasp):
            self.open_gripper(dt=dt)
        elif isinstance(command, Sequence):
            for subcommand in command.commands:
                self.execute_command(subcommand, dt=dt)
        else:
            raise ValueError("Unknown command type: {}".format(command))


@dataclass
class Command:
    """A command that can be executed in the environment."""

    def __repr__(self):
        return self.__class__.__name__


@dataclass
class Trajectory(Command):
    robot: int
    joints: List[int]
    path: List[List[float]]
    time_after_contact: float = np.inf
    attachments: List[Attachment] = field(default_factory=list)

    def reverse(self):
        return self.__class__(
            self.robot,
            self.joints,
            self.path[::-1],
            time_after_contact=self.time_after_contact,
            attachments=self.attachments,
        )

    def __repr__(self):
        return "t{}".format(id(self) % 1000)


@dataclass
class ActivateGrasp(Command):
    """Create or remove a fixed joint between two bodies and open/close a
    gripper."""

    robot: int
    gripper_link: int
    body: int

    def __repr__(self):
        return "s{}".format(id(self) % 1000)


@dataclass
class DeactivateGrasp(Command):
    """Create or remove a fixed joint between two bodies and open/close a
    gripper."""

    robot: int
    gripper_link: int
    body: int

    def __repr__(self):
        return "s{}".format(id(self) % 1000)


@dataclass
class Sequence(Command):
    """A named sequence of commands to execute."""

    commands: List[Command]
    name: str = None

    def __repr__(self):
        class_name = self.__class__.__name__
        sequence_name = f" {self.name}" if hasattr(self, "name") else ""
        commands_repr = " ".join(
            [
                (
                    repr(command)
                    if not isinstance(command, Sequence)
                    else repr(command).replace("(", "").replace(")", "")
                )
                for command in self.commands
            ]
        )
        return f"{class_name}{sequence_name}({commands_repr})"


@dataclass
class Conf:
    robot: int
    joints: List[int]
    positions: List[float]

    def __repr__(self):
        return "q{}".format(id(self) % 1000)

    def assign(self, sim: SimulatorInstance):
        pbu.set_joint_positions(
            sim.robot, self.joints, self.positions, client=sim.client
        )


@dataclass
class Attachment:
    parent: int
    parent_link: int
    child: int
    parent_T_child: pbu.Pose

    def __repr__(self):
        return "a{}".format(id(self) % 1000)


@dataclass
class Grasp:
    attachment: Attachment
    closed_position: float = 0.0

    def get_pregrasp_pose(
        self,
        grasp_pose: pbu.Pose,
        gripper_T_tool: pbu.Pose = pbu.unit_pose(),
        tool_distance: float = PREGRASP_DISTANCE,
        object_distance: float = PREGRASP_DISTANCE,
    ) -> pbu.Pose:
        return pbu.multiply(
            gripper_T_tool,
            pbu.Pose(pbu.Point(x=tool_distance)),
            grasp_pose,
            pbu.Pose(pbu.Point(z=-object_distance)),
        )

    def __repr__(self):
        return "g{}".format(id(self) % 1000)
