import os


def get_root_path():
    current_path = os.path.abspath(os.path.dirname(__file__))
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
    return os.path.join(root_path, "xbot")


def get_config_path():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config"))
    return config_path


def get_data_path():
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/")
    )
    return config_path
