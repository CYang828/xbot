import os


def get_root_path():
    current_path = os.path.abspath(os.path.dirname(__file__))
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
    return os.path.join(root_path, 'xbot')
