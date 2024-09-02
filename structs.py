
import pb_utils as pbu
import numpy as np
import pybullet_utils.bullet_client as bc
import itertools
import pybullet as p
from typing import Dict, List
from dataclasses import dataclass
from abc import ABC
from dataclasses import field

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
    mesh: str # Path to .obj
    pose: pbu.Pose # Current pose of the object
    category: str = "" # Optional class label or object name

@dataclass
class WorldBelief:
    object_states: List[ObjectState] # Mapping from id in pybullet client to object state
    robot_state: List[float] # Current robot movable joint positions

@dataclass
class SimulatorInstance:
    client: int # The pybullet phyiscs client
    robot: int # The robot id in the pybullet client
    table: int # The table id in the pybullet client
    movable_objects: Dict[int, ObjectState] # Mapping from id in pybullet client to object state
    components: Dict[str, int] = field(default_factory=dict) # Mapping from group name to component id

    @staticmethod
    def from_belief(belief: WorldBelief, gui=False):
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
        movable_objects = {}
        for obj_id, obj_state in belief.object_states.items():
            obj = pbu.load_pybullet(obj_state.mesh, client=client)
            pbu.set_pose(obj, obj_state.pose, client=client)
            movable_objects[obj] = obj_state
            
        return SimulatorInstance(client, robot_body, table, movable_objects)
    

    def get_group_subtree(self, group, **kwargs):
        return pbu.get_link_subtree(
            self.body, self.get_group_parent(group, **kwargs), **kwargs
        )

    def get_component(self, group, visual=True, **kwargs):
        if group not in self.components:
            component = pbu.clone_body(
                self.body,
                links=self.get_group_subtree(group, **kwargs),
                visual=False,
                collision=True,
                **kwargs,
            )
            if not visual:
                pbu.set_all_color(component, pbu.TRANSPARENT)
            self.components[group] = component
        return self.components[group]
    
    @property
    def tool_link(self):
        pbu.link_from_name(self.robot, PANDA_TOOL_TIP, client=self.client)
    
    def get_group_joints(self, group, **kwargs):
        return pbu.joints_from_names(
            self.robot, PANDA_GROUPS[group], **kwargs
        )

class SimulatorState:
    def __init__(self, instance, attachments={}):
        super(SimulatorState, self).__init__(attachments)
        self.attachments = attachments
        self.instance = instance

    def propagate(self, **kwargs):
        for relative_pose in self.attachments.values():
            relative_pose.assign(**kwargs)


class Conf(object):
    def __init__(self, robot, joints, positions):
        self.robot = robot
        self.joints = joints
        self.positions = positions

    @property
    def values(self):
        return self.positions

    def assign(self):
        pbu.set_joint_positions(self.body, self.joints, self.positions)

    def iterate(self):
        yield self

    def __repr__(self):
        return "q{}".format(id(self) % 1000)
    
class GroupConf(Conf):
    def __init__(self, body, group, *args, **kwargs):
        joints = body.get_group_joints(group, **kwargs)
        super(GroupConf, self).__init__(body, joints, *args, **kwargs)
        self.group = group

    def __repr__(self):
        return "{}q{}".format(self.group[0], id(self) % 1000)
    
class Command(ABC):

    def iterate(self, state, **kwargs):
        raise NotImplementedError()

    def controller(self, *args, **kwargs):
        raise NotImplementedError()

    def execute(self, controller, *args, **kwargs):
        return True

    
class Trajectory(Command):
    def __init__(
        self,
        sim:SimulatorInstance,
        joints,
        path,
        velocity_scale=1.0,
        contact_links=[],
        time_after_contact=np.inf,
        contexts=[],
        **kwargs,
    ):
        self.sim = sim
        self.joints = joints
        self.path = tuple(path)
        self.velocity_scale = velocity_scale
        self.contact_links = tuple(contact_links)
        self.time_after_contact = time_after_contact
        self.contexts = tuple(contexts)

    @property
    def context_bodies(self):
        return {self.body} | {context.body for context in self.contexts}

    def conf(self, positions):
        return Conf(self.body, self.joints, positions=positions)

    def first(self):
        return self.conf(self.path[0])

    def last(self):
        return self.conf(self.path[-1])

    def reverse(self):
        return self.__class__(
            self.sim,
            self.joints,
            self.path[::-1],
            velocity_scale=self.velocity_scale,
            contact_links=self.contact_links,
            time_after_contact=self.time_after_contact,
            contexts=self.contexts,
        )

    def adjust_path(self, **kwargs):
        current_positions = pbu.get_joint_positions(
            self.body, self.joints, **kwargs
        )
        return pbu.adjust_path(
            self.body, self.joints, [current_positions] + list(self.path), **kwargs
        )

    def compute_waypoints(self, **kwargs):
        return pbu.waypoints_from_path(
            pbu.adjust_path(self.body, self.joints, self.path, **kwargs)
        )

    def compute_curve(self, **kwargs):
        path = self.adjust_path(**kwargs)
        positions_curve = pbu.interpolate_path(self.body, self.joints, path, **kwargs)
        return positions_curve

    def iterate(self, state, teleport=False, **kwargs):
        if teleport:
            pbu.set_joint_positions(self.body, self.joints, self.path[-1], **kwargs)
            return self.path[-1]
        else:
            return pbu.step_curve(self.body, self.joints, self.compute_curve(**kwargs), **kwargs)

    def __repr__(self):
        return "t{}".format(id(self) % 1000)


class CaptureImage(Command):
    def __init__(self, robot=None, captured_image=None, **kwargs):
        self.robot = robot
        self.captured_image = captured_image

    def iterate(self, state, **kwargs):
        self.captured_image = self.robot.get_image()
        return pbu.empty_sequence()


class GroupTrajectory(Trajectory):
    def __init__(self, sim:SimulatorInstance, group:str, path, *args, **kwargs):
        self.sim = sim
        joints = self.sim.get_group_joints(group, **kwargs)
        super(GroupTrajectory, self).__init__(self.sim, joints, path, *args, **kwargs)
        self.group = group

    def reverse(self, **kwargs):
        return self.__class__(
            self.body,
            self.group,
            self.path[::-1],
            velocity_scale=self.velocity_scale,
            contact_links=self.contact_links,
            time_after_contact=self.time_after_contact,
            contexts=self.contexts,
            **kwargs,
        )

    def __repr__(self):
        return "{}t{}".format(self.group[0], id(self) % 1000)



class ParentBody(object):
    def __init__(self, sim:SimulatorInstance, link=pbu.BASE_LINK):
        self.body = sim.robot
        self.link = link

    def __iter__(self):
        return iter([self.body, self.link])

    def get_pose(self, **kwargs):
        if self.body is None:
            return pbu.unit_pose()
        return pbu.get_link_pose(self.body, self.link, **kwargs)

    def __repr__(self):
        return "Parent({})".format(self.body)
    


class Switch(Command):
    def __init__(self, sim:SimulatorInstance, parent=None):
        self.sim = sim
        self.parent = parent

    def iterate(self, state, **kwargs):

        if self.parent is None and self.body in state.attachments.keys():
            del state.attachments[self.body]

        elif self.parent is not None:
            robot, _ = self.parent

            gripper_joints = robot.get_group_joints(GRIPPER_GROUP, **kwargs)
            finger_links = robot.get_finger_links(gripper_joints, **kwargs)

            movable_bodies = [
                body for body in pbu.get_bodies(**kwargs) if (body != robot)
            ]

            max_distance = 5e-2
            collision_bodies = [
                body
                for body in movable_bodies
                if (
                    all(
                        pbu.get_closest_points(
                            robot, body, link1=link, max_distance=max_distance, **kwargs
                        )
                        for link in finger_links
                    )
                    and pbu.get_mass(body, **kwargs) != pbu.STATIC_MASS
                )
            ]

            if len(collision_bodies) > 0:
                relative_pose = RelativePose(
                    collision_bodies[0], parent=self.parent, **kwargs
                )
                state.attachments[self.body] = relative_pose

        return pbu.empty_sequence()

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.body)

    def to_lisdf(self):
        return []
    

class Grasp(object):  # RelativePose
    def __init__(self, sim:SimulatorInstance, grasp, pregrasp=None, closed_position=0.0, **kwargs):
        self.sim = sim
        self.grasp = grasp
        if pregrasp is None:
            pregrasp = self.get_pregrasp(grasp)
        self.pregrasp = pregrasp
        self.closed_position = closed_position

    def get_pregrasp(
        self,
        grasp_tool,
        gripper_from_tool=pbu.unit_pose(),
        tool_distance=PREGRASP_DISTANCE,
        object_distance=PREGRASP_DISTANCE,
    ):
        return pbu.multiply(
            gripper_from_tool,
            pbu.Pose(pbu.Point(x=tool_distance)),
            grasp_tool,
            pbu.Pose(pbu.Point(z=-object_distance)),
        )


    @property
    def value(self):
        return self.grasp

    @property
    def approach(self):
        return self.pregrasp

    def create_relative_pose(
        self, robot, link=pbu.BASE_LINK, **kwargs
    ):
        parent = ParentBody(body=robot, link=link, **kwargs)
        return RelativePose(
            self.body, parent=parent, relative_pose=self.grasp, **kwargs
        )

    def create_attachment(self, *args, **kwargs):
        relative_pose = self.create_relative_pose(*args, **kwargs)
        return relative_pose.get_attachment()

    def __repr__(self):
        return "g{}".format(id(self) % 1000)
    
class Sequence(Command):
    def __init__(self, commands=[], name=None):
        self.context = None
        self.commands = tuple(commands)
        self.name = self.__class__.__name__.lower()[:3] if (name is None) else name

    @property
    def context_bodies(self):
        return set(
            itertools.chain(*[command.context_bodies for command in self.commands])
        )

    def __len__(self):
        return len(self.commands)

    def iterate(self, *args, **kwargs):
        for command in self.commands:
            print("Executing {} command: {}".format(type(command), str(command)))
            for output in command.iterate(*args, **kwargs):
                yield output

    def controller(self, *args, **kwargs):
        return itertools.chain.from_iterable(
            command.controller(*args, **kwargs) for command in self.commands
        )

    def execute(self, *args, return_executed=False, **kwargs):
        executed = []
        for command in self.commands:
            if not command.execute(*args, **kwargs):
                return False, executed if return_executed else False
            executed.append(command)
        return True, executed if return_executed else True

    def reverse(self):
        return Sequence(
            [command.reverse() for command in reversed(self.commands)], name=self.name
        )

    def dump(self):
        print("[{}]".format(" -> ".join(map(repr, self.commands))))

    def __repr__(self):
        return "{}({})".format(self.name, len(self.commands))
    

class RelativePose(object):
    def __init__(
        self,
        sim:SimulatorInstance,
        body,
        parent=None,
        parent_state=None,
        relative_pose=None,
        important=False,
        **kwargs,
    ):
        self.sim = sim
        self.body = body
        self.parent = parent
        self.parent_state = parent_state
        if relative_pose is None:
            relative_pose = pbu.multiply(
                pbu.invert(self.get_parent_pose(**kwargs)),
                pbu.get_pose(self.body, **kwargs),
            )
        self.relative_pose = tuple(relative_pose)
        self.important = important

    @property
    def value(self):
        return self.relative_pose

    def ancestors(self):
        if self.parent_state is None:
            return [self.body]
        return self.parent_state.ancestors() + [self.body]

    def get_parent_pose(self, **kwargs):
        if self.parent is None:
            return pbu.unit_pose()
        if self.parent_state is not None:
            self.parent_state.assign(**kwargs)
        return self.parent.get_pose(**kwargs)

    def get_pose(self, **kwargs):
        return pbu.multiply(self.get_parent_pose(**kwargs), self.relative_pose)

    def assign(self, **kwargs):
        world_pose = self.get_pose(**kwargs)
        pbu.set_pose(self.body, world_pose, **kwargs)
        return world_pose

    def get_attachment(self, **kwargs):
        assert self.parent is not None
        parent_body, parent_link = self.parent
        return pbu.Attachment(
            parent_body, parent_link, self.relative_pose, self.body, **kwargs
        )

    def __repr__(self):
        name = "wp" if self.parent is None else "rp"
        return "{}{}".format(name, id(self) % 1000)
   