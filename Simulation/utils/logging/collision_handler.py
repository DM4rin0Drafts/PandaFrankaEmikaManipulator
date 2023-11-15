import logging

from Simulation.utils.logging.runtime_handler import LoggingContext, MODE


def debug_collision_links(body1, body2, collisions):
    """
    Save all collisions notifications in a file

    Parameters
    ----------
        body1: int, required
            the body unique id, as returned by loadURDF etc.
        body2: int, required
            the body unique id, as returned by loadURDF etc.
        collisions: list, required
            list full of joint ids that collides in the simulation
    """
    logger = logging.getLogger(MODE[0])
    msg = "Collision happen for bodies: {}, {}\n".format(body1, body2)
    collisions_msg = ""

    for c in collisions:
        collisions_msg = collisions_msg + "collision in joint {} and {}\n".format(
            c[0], c[1])

    with LoggingContext(logger, level=logging.DEBUG):
        logger.debug(msg + collisions_msg)
