import os
import yaml
from box import ConfigBox
from pathlib import Path
from accord import logger
from box.exceptions import BoxValueError



CONFIG_PATH = Path("configs/tools_config.yaml")

def read_yaml(path_to_yaml:Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

def get_config(CONFIG_PATH=CONFIG_PATH):
    """returns ConfigBox type

    Returns:
        ConfigBox: ConfigBox type
    """
    return read_yaml(CONFIG_PATH)


def create_directory(directory_path: str) -> None:
    """
    Create a directory if it does not exist.

    Parameters:
        directory_path (str): The path of the directory to be created.

    Example:
    ```python
    create_directory("/path/to/new/directory")
    ```

    """
    os.makedirs(directory_path, exist_ok=True)
    logger.info(f"Directory '{directory_path}' was created.")