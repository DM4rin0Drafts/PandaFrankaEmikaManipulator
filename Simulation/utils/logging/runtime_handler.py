from Simulation.utils.pybullet_tools.os_utils import get_real_path

import logging, sys


# Logging Massages
# Info	   Everything is fine and program can run this part of program
# Debug	   Detailed program message for debugging module
# Warning  Program runs normal, but a unexpected situation happens
#          (i.e. program run out of memory)
# Error	   Error of a module, cannot run the program
# Critical  Critical Error happened

# function_name=None, path_of_error=None, model=None, joint_name=None


MODE = {
    None: 'root',
    0: 'HELP',
    1: 'SIM',
    2: 'KIN',
    3: 'DYN',
    4: 'JOINTS'
}


class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        """
        Enable flags for loading model into pybullet simulation

        Parameters
        ----------
            logger: bool, optional
                Enable/Disable URDF graphics shapes
            level:

            handler:

            close:

        """
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()


# CONFIGURATION
filename = get_real_path("/logs/run_time.log")
log_format = ('[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s')


def set_logger_config(level):
    """
    Enable flags for loading model into pybullet simulation

    Parameters
    ----------
        level: bool, optional
            Enable/Disable URDF graphics shapes

    return
    ----------
        Simulation flags (see comments which already initialized)

    """
    logging.basicConfig(level=level, format=log_format, filename=filename,
                        filemode='w')


def save_msg_in_log_file(msg, root, log_func):
    """
    Enable flags for loading model into pybullet simulation

        Parameters
        ----------
            msg: bool, optional
                Enable/Disable URDF graphics shapes
            root:

            log_func:

    """
    logger = logging.getLogger(root)

    if log_func.__name__ == 'info':
        logger.info(msg.upper())
    elif log_func.__name__ == 'warning':
        logger.warning(msg)
    elif log_func.__name__ == 'error':
        logger.error(msg)


# Save messages in log file
def info_logger(msg, root=None):
    """
    Enable flags for loading model into pybullet simulation

        Parameters
        ----------
            msg: bool, optional
                Enable/Disable URDF graphics shapes
            root:

    """
    # msg = message, root = categories [SIM (SIMULATION),
    #                                   KIN (KINEMATIC),
    #                                   DYN (DYNAMIC), HELP]
    set_logger_config(logging.INFO)
    save_msg_in_log_file(msg, MODE[root], logging.info)


def warning_logger(msg, root=None):
    """
    Enable flags for loading model into pybullet simulation

    Parameters
    ----------
        msg: bool, optional
            Enable/Disable URDF graphics shapes
        root:

    """
    set_logger_config(logging.WARNING)
    save_msg_in_log_file(msg, MODE[root], logging.warning)


def error_logger(msg, root=None):
    """
    Enable flags for loading model into pybullet simulation

    Parameters
    ----------
        msg: bool, optional
            Enable/Disable URDF graphics shapes
        root:

    """
    set_logger_config(logging.ERROR)
    save_msg_in_log_file(msg, MODE[root], logging.error)


# DEBUGGING
def debug_load_model(function, function_path, filename):
    """
    Enable flags for loading model into pybullet simulation

    Parameters
    ----------
        function: bool, optional
            Enable/Disable URDF graphics shapes
        function_path:

        filename:

    """
    logger = logging.getLogger(MODE[0])
    msg = "In Function [{}] in: {}\n\nCheck filename-path {}\n\nLook in " \
          "file Simulation/Environment/build_scenario_*.py " \
          "or Simulation/utils/pybullet_tools/utils and change function [get_real_path]".format(
        function, function_path, filename)

    with LoggingContext(logger, level=logging.DEBUG):
        logger.debug(msg)


def debug_joint_config(model, model_name, joint_id, joint_name,
                       limits, current_position):
    """
    Enable flags for loading model into pybullet simulation

    Parameters
    ----------
        model: bool, optional
            Enable/Disable URDF graphics shapes
        model_name:

        joint_id:

        joint_name:

        limits:

        current_position:

    """
    logger = logging.getLogger(MODE[0])
    model = "Model: [{}, {}] has the follwing Limits: [{}, {}] for the current joint  [{}, {}] \n".format(
        model, model_name, limits[0], limits[1], joint_id, joint_name)
    config = "Current position is {}\n\n".format(current_position)

    with LoggingContext(logger, level=logging.DEBUG):
        logger.debug(model + config)
