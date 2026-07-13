import yaml
from typing import Any

def read_yaml_file(filepath: str) -> Any:
    """
    Reads a YAML file from the given filepath and returns its content as a dictionary or other loaded object.
    """
    try:
        with open(filepath, 'r') as file:
            content = yaml.safe_load(file)
        return content
    except FileNotFoundError:
        return f"Error: File not found at {filepath}"
    except yaml.YAMLError as e:
        return f"Error parsing YAML file: {e}"