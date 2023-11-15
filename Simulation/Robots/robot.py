from Simulation.utils.pybullet_tools.pybullet_utils import get_all_joint_limits, get_all_limbs_limits, get_limb_joints, get_joints, \
    get_position, get_all_max_velocities, get_all_max_forces, get_real_path
from Simulation.Robots.robot_utils import ROBOT_GROUPS_ID, PANDA_ACCELERATION

from roboticstoolbox import RevoluteDH, DHRobot, RevoluteMDH
from roboticstoolbox.robot.ERobot import ERobot

from spatialmath.base import trotz, transl
from spatialmath import SE3
from numpy import pi
import numpy as np


class RobotConfig(object):
    def __init__(self, body, name, dof, custom_limits=None):
        """
        Initialize all robot information. All information are saved as numpy arrays

        Parameters
        ----------
            body: int, required
                Individual pybullet id of the model
            name: str, required
                Name of the robot model
            dof: int, required
                Degrees of freedom of the robot model
            custom_limits: list/numpy, required
                Set custom limits to the robot model, instead of using true limits
        """
        self.body = body
        self.name = name

        if isinstance(custom_limits, list) or isinstance(custom_limits, np.ndarray):
            self.joint_limits = np.asarray(custom_limits)
        else:
            self.joint_limits = np.asarray(get_all_limbs_limits(self.body))
        
        # degrees of freedom
        self.dof = dof

        # joint limb of the robot model. gripper and joint ids, without finger2, finger3, ...
        self.joints = get_limb_joints()
        # all joint id's
        self.movable_joints = get_joints(self.body)[:dof]
        # gripper id's
        self.movable_gripper = ROBOT_GROUPS_ID['gripper']
        # position of the robot base [X, Y, Z]
        self.base_position = np.array(get_position(self.body))
        # Maximum Velocity with a shape of (dof,)
        self.max_velocity = np.array(get_all_max_velocities(self.body, self.movable_joints))
        # Maximum forces with a shape of (dof,)
        self.max_forces = np.array(get_all_max_forces(self.body, self.movable_joints))
        # Maximum acceleration with a shape of (dof,)
        self.max_acceleration = np.array(PANDA_ACCELERATION)


class RobotSetup(RobotConfig):
    def __init__(self, body, name, dof=7, file_name=None, custom_limits=None):
        """
        Load and set robot model.

        Parameters
        ----------
            body: int, required
                Individual pybullet id of the model
            name: str, required
                Name of the robot model
            dof: int, required
                Degrees of freedom of the robot model
            file_name: str, optional
                Path to the filename that are loaded into the environment
            custom_limits: list/numpy, optional
                Set custom limits to the robot model, instead of using true limits. 
                Input shape for costum limits needs to be (dof, 2)
        """
        super().__init__(body, name, dof, custom_limits)

        if isinstance(file_name, str):
            links, name, urdf_string, urdf_filepath = ERobot.URDF_read(file_path=get_real_path(file_name))

            # https://petercorke.github.io/robotics-toolbox-python/arm_dh.html?highlight=dhrobot#roboticstoolbox.robot.DHRobot.DHRobot.rne_python
            self.model = ERobot(links, name=name,
                                manufacturer="Franka Emika",
                                gripper_links=links[9],  # link 9
                                urdf_string=urdf_string,
                                urdf_filepath=urdf_filepath)

            self.model.grippers[0].tool = SE3(0, 0, 0.1034)
            self.model.qr = np.array([0., -0.3, 0., -2.2, 0., 2., 0.78539816])
        else:
            raise ValueError("file_name is not a string")


class Robot(DHRobot):
    def __init__(self, body, dh_table, tool=None, joint_config=None):
        """
        Load and set robot model.

        Parameters
        ----------
            body: int, required
                Individual pybullet id of the model
            dh_table: str, required
                Name of the robot model
            tool: int, required
                Degrees of freedom of the robot model
            joint_config: str, optional
                Path to the filename that are loaded into the environment
                Example for joint_config = [("qz", [0, 0, 0, 0, 0, 0]) , ("qr", [0, -0.3, -2.2, 0, 2.0, np.pi/4])]
        """
        super().__init__(dh_table, tool=tool)

        self.joint_limits = get_all_joint_limits(self.body)
        self.body = body
        if joint_config is not None:
            for config in joint_config:
                self.addconfiguration(config[0], config[1])


# https://petercorke.github.io/robotics-toolbox-python/_modules/roboticstoolbox/models/URDF/Panda.html
# alternative rtb.models.DH.Panda()
class Panda(DHRobot, RobotConfig):
    def __init__(self, body, joint_config=None):
        """
        Load DH parameter PANDA robot model with the RTB-Library.

        Parameters
        ----------
            body: int, required
                Individual pybullet id of the model
            joint_config: str, optional
                Add standard joint positions to the robot model
        """
        tool = transl(0, 0, 0.103) @  trotz(-pi/4)
        super().__init__([RevoluteMDH(d=0.333, qlim=[-2.8973, 2.8973]),
                          RevoluteMDH(alpha=-pi/2, qlim=[-1.7628, 1.7628]),
                          RevoluteMDH(d=0.316, alpha=pi/2, qlim=[-2.8973, 2.8973]),
                          RevoluteMDH(a=0.0825, alpha=pi/2, qlim=[-3.0718, -0.0698]),
                          RevoluteMDH(a=-0.0825, d=0.384, alpha=-pi/2, qlim=[-2.8973, 2.8973]),
                          RevoluteMDH(alpha=pi/2, qlim=[-0.0175, 3.7525]),
                          RevoluteMDH(a=0.088, d=0.107, alpha=pi/2, qlim=[-2.8973, 2.8973])], name='Panda', tool=tool)
        self.body = body

        if joint_config is not None:
            for config in joint_config:
                self.addconfiguration(config[0], config[1])

        self.set_joint_limits(self.body)


# alternative rtb.models.DH.Puma560()
class Puma560(DHRobot):
    """
    Load DH parameter PUMA robot model with the RTB-Library.

    Parameters
    ----------
        joint_config: str, optional
            Add standard joint positions to the robot model
    """
    def __init__(self, joint_config=None):
        super().__init__([RevoluteDH(alpha=pi/2, d=0.6718, qlim=[-2.7925, 2.7925], Jm=4),
                          RevoluteDH(a=0.4318, qlim=[-11/18*pi, 11/18*pi]),
                          RevoluteDH(d=0.15005, a=0.0203, alpha=-pi/2, qlim=[-3/4*pi, 3/4*pi]),
                          RevoluteDH(d=0.4318, alpha=pi/2, qlim=[-133/90*pi, 133/90*pi]),
                          RevoluteDH(alpha=-pi/2, qlim=[-5/9*pi, 5/9*pi]),
                          RevoluteDH(qlim=[-133/90*pi, 133/90*pi])], name="Puma560",)
        
        if joint_config is not None:
            for config in joint_config:
                self.addconfiguration(config[0], config[1])
