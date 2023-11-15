from Simulation.utils.pybullet_tools.pybullet_utils import HideOutput, LockRenderer, load_pybullet_model, \
    set_position, activate_gravity
from Simulation.Robots.robot_utils import get_initial_grasp_type, set_arm_config, open_gripper
from Simulation.utils.logging.runtime_handler import info_logger


class BuildWorld(object):
    def __init__(self, place_obstacle=False):
        """
        Build pybullet environment.

        Parameters
        ----------
            place_obstacle: bool, optional
                place an obstacle in the simulation for collision testing task
        """
        with HideOutput():
            with LockRenderer():
                info_logger("start simulation\n")

                self.table_config = [1.5, 1, 0.68]

                info_logger("LOAD MODELS", 1)
                # load FLOOR model to environment
                floor_path = '/Simulation/utils/models/floor.urdf'
                self.floor = load_pybullet_model(floor_path, fixed_base=True)

                # load TABLE model to environment
                table_path = '/Simulation/utils/models/table_collision/table.urdf'
                self.table = load_pybullet_model(table_path, fixed_base=True)

                # load ROBOT model to environment
                self.robot_path = '/Simulation/Robots/panda/panda.urdf'
                self.robot = load_pybullet_model(self.robot_path, fixed_base=True)

                # load ROBOT model to environment
                if place_obstacle:
                    box_path = '/Simulation/utils/models/box.urdf'
                    self.box = load_pybullet_model(box_path, fixed_base=False, scale=5)

                info_logger('Models successful loaded\n\n', 1)

                self.configuration(place_obstacle)
                activate_gravity()

        if place_obstacle:
            self.movable_bodies = [self.box]
        else:
            self.movable_bodies = []
        
        self.env_bodies = [self.floor]
        self.regions = [self.table]
        self.robots = [self.robot]

        self.all_bodies = [
            set(self.movable_bodies) | set(self.env_bodies) | set(
                self.regions)]

    def configuration(self, place_obstacle):
        """
        Description: This functions configure the position and orientation of the spawned bodies
        """
        # Set start position and joint angles of the robot
        if place_obstacle:
            set_position(self.box, [-.2, 0.4, 0.88])
        
        set_position(self.robot, [-0.4, 0, self.table_config[2] - 0.05])  # [x, y, z]-Position
        arm_config = get_initial_grasp_type('qr')

        set_arm_config(self.robot, "arm", arm_config)
        open_gripper(self.robot)
