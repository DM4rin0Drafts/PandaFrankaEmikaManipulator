import platform
import os


def get_operating_system():
    """
    Return the operating system that are currently used for the program
    """
    return platform.system()


def up_directory(path):
    """
    Return path and go one directory up

    Parameters
    ----------
        path: str, required
            Full path of file/directory

    """
    return os.path.dirname(path)


def get_real_path(path):
    """
    Return the full path of a file

    Parameters
    ----------
        path: str, required
            Path of a file or directory

    """
    operating_system = get_operating_system()

    if operating_system == "Windows":
        direction = up_directory(
            up_directory(up_directory(up_directory(os.path.abspath(__file__))))
        )
        path = os.path.abspath(path).replace("C:", '')

        # Path has to change in Windows-file-system-format
        return direction + path
    else:
        # Note for MacOS it works
        direction = up_directory(up_directory(
            up_directory(up_directory(os.path.abspath(__file__)))
        ))
        return direction + path
