import os
import uuid
import string
from pathlib import Path
from typing import Text, List


import xbot.util.config

import pytest


os.environ["USER_NAME"] = "user"
os.environ["PASS"] = "pass"


def test_read_yaml_string():
    config_without_env_var = """
    user: user
    password: pass
    """
    content = xbot.util.config.read_yaml(config_without_env_var)
    assert content["user"] == "user" and content["password"] == "pass"


def test_read_yaml_string_with_env_var():
    config_with_env_var = """
    user: ${USER_NAME}
    password: ${PASS}
    """
    content = xbot.util.config.read_yaml(config_with_env_var)
    assert content["user"] == "user" and content["password"] == "pass"


def test_read_yaml_string_with_multiple_env_vars_per_line():
    config_with_env_var = """
    user: ${USER_NAME} ${PASS}
    password: ${PASS}
    """
    content = xbot.util.config.read_yaml(config_with_env_var)
    assert content["user"] == "user pass" and content["password"] == "pass"


def test_read_yaml_string_with_env_var_prefix():
    config_with_env_var_prefix = """
    user: db_${USER_NAME}
    password: db_${PASS}
    """
    content = xbot.util.config.read_yaml(config_with_env_var_prefix)
    assert content["user"] == "db_user" and content["password"] == "db_pass"


def test_read_yaml_string_with_env_var_postfix():
    config_with_env_var_postfix = """
    user: ${USER_NAME}_admin
    password: ${PASS}_admin
    """
    content = xbot.util.config.read_yaml(config_with_env_var_postfix)
    assert content["user"] == "user_admin" and content["password"] == "pass_admin"


def test_read_yaml_string_with_env_var_infix():
    config_with_env_var_infix = """
    user: db_${USER_NAME}_admin
    password: db_${PASS}_admin
    """
    content = xbot.util.config.read_yaml(config_with_env_var_infix)
    assert content["user"] == "db_user_admin" and content["password"] == "db_pass_admin"


def test_read_yaml_string_with_env_var_not_exist():
    config_with_env_var_not_exist = """
    user: ${USER_NAME}
    password: ${PASSWORD}
    """
    with pytest.raises(ValueError):
        xbot.util.config.read_yaml(config_with_env_var_not_exist)


def test_environment_variable_not_existing():
    content = "model: \n  test: ${variable}"
    with pytest.raises(ValueError):
        xbot.util.config.read_yaml(content)


def test_environment_variable_dict_without_prefix_and_postfix():
    os.environ["variable"] = "test"
    content = "model: \n  test: ${variable}"

    content = xbot.util.config.read_yaml(content)

    assert content["model"]["test"] == "test"


def test_environment_variable_in_list():
    os.environ["variable"] = "test"
    content = "model: \n  - value\n  - ${variable}"

    content = xbot.util.config.read_yaml(content)

    assert content["model"][1] == "test"


def test_environment_variable_dict_with_prefix():
    os.environ["variable"] = "test"
    content = "model: \n  test: dir/${variable}"

    content = xbot.util.config.read_yaml(content)

    assert content["model"]["test"] == "dir/test"


def test_environment_variable_dict_with_postfix():
    os.environ["variable"] = "test"
    content = "model: \n  test: ${variable}/dir"

    content = xbot.util.config.read_yaml(content)

    assert content["model"]["test"] == "test/dir"


def test_environment_variable_dict_with_prefix_and_with_postfix():
    os.environ["variable"] = "test"
    content = "model: \n  test: dir/${variable}/dir"

    content = xbot.util.config.read_yaml(content)

    assert content["model"]["test"] == "dir/test/dir"


def test_emojis_in_yaml():
    test_data = """
    data:
        - one 😁💯 👩🏿‍💻👨🏿‍💻
        - two £ (?u)\\b\\w+\\b f\u00fcr
    """
    content = xbot.util.config.read_yaml(test_data)

    assert content["data"][0] == "one 😁💯 👩🏿‍💻👨🏿‍💻"
    assert content["data"][1] == "two £ (?u)\\b\\w+\\b für"


def test_emojis_in_tmp_file():
    test_data = """
        data:
            - one 😁💯 👩🏿‍💻👨🏿‍💻
            - two £ (?u)\\b\\w+\\b f\u00fcr
        """
    test_file = xbot.util.io.create_temporary_file(test_data)
    content = xbot.util.config.read_yaml_file(test_file)

    assert content["data"][0] == "one 😁💯 👩🏿‍💻👨🏿‍💻"
    assert content["data"][1] == "two £ (?u)\\b\\w+\\b für"


def test_read_emojis_from_json():
    import json

    d = {"text": "hey 😁💯 👩🏿‍💻👨🏿‍💻🧜‍♂️(?u)\\b\\w+\\b} f\u00fcr"}
    json_string = json.dumps(d, indent=2)

    content = xbot.util.config.read_yaml(json_string)

    expected = "hey 😁💯 👩🏿‍💻👨🏿‍💻🧜‍♂️(?u)\\b\\w+\\b} für"
    assert content.get("text") == expected


def test_bool_str():
    test_data = """
    one: "yes"
    two: "true"
    three: "True"
    """

    content = xbot.util.config.read_yaml(test_data)

    assert content["one"] == "yes"
    assert content["two"] == "true"
    assert content["three"] == "True"


@pytest.mark.parametrize(
    "should_preserve_key_order, expected_keys",
    [(True, list(reversed(string.ascii_lowercase)))],
)
def test_dump_yaml_key_order(
    tmp_path: Path, should_preserve_key_order: bool, expected_keys: List[Text]
):
    file = tmp_path / "test.yml"

    # create YAML file with keys in reverse-alphabetical order
    content = ""
    for i in reversed(string.ascii_lowercase):
        content += f"{i}: {uuid.uuid4().hex}\n"

    file.write_text(content)

    # load this file and ensure keys are in correct reverse-alphabetical order
    data = xbot.util.config.read_yaml_file(file)
    assert list(data.keys()) == list(reversed(string.ascii_lowercase))

    # dumping `data` will result in alphabetical or reverse-alphabetical list of keys,
    # depending on the value of `should_preserve_key_order`
    xbot.util.config.write_yaml(
        data, file, should_preserve_key_order=should_preserve_key_order
    )
    with file.open() as f:
        keys = [line.split(":")[0] for line in f.readlines()]

    assert keys == expected_keys
