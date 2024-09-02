import pickle
import zlib

import zmq
import zmq.ssh


class PandaSender:
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:5554")

    def capture_realsense(self):
        print("[Controller] Capturing realsense")
        self.socket.send(
            zlib.compress(pickle.dumps({"message_name": "capture_realsense"}))
        )
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message["rgb"], message["depth"], message["intrinsics"]

    def command_arm(self, positions):
        print("[Controller] Commanding arm to target config")
        self.socket.send(
            zlib.compress(
                pickle.dumps({"message_name": "command_arm", "positions": positions})
            )
        )
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def get_joint_states(self):
        print("[Controller] Getting joint states")
        self.socket.send(
            zlib.compress(pickle.dumps({"message_name": "get_joint_states"}))
        )
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message["joint_states"]

    def open_gripper(self):
        print("[Controller] Opening gripper")
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "open_gripper"})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def close_gripper(self):
        print("[Controller] Closing gripper")
        self.socket.send(zlib.compress(pickle.dumps({"message_name": "close_gripper"})))
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message

    def execute_position_path(self, pdicts):
        print("[Controller] Executing position path")
        self.socket.send(
            zlib.compress(
                pickle.dumps(
                    {"message_name": "execute_position_path", "pdicts": pdicts}
                )
            )
        )
        message = pickle.loads(zlib.decompress(self.socket.recv()))
        return message
