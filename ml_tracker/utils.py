from pathlib import Path
import json


def write_json(path: Path | str, info: dict) -> bool:
    """
    Write a dictionary to a JSON file.

    Args:
        path (str or Path): Path to the JSON file.
        info (dict): Dictionary to write.

    Returns:
        bool: True if write succeeds.

    Raises:
        TypeError: If `info` is not a dictionary.
        IOError: If file cannot be written.
    """
    if not isinstance(info, dict):
        raise TypeError("info must be a dictionary")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
    return True


def read_json(path: Path | str) -> dict:
    """
    Read a JSON file and return its contents as a dictionary.

    Args:
        path (str or Path): Path to the JSON file.

    Returns:
        dict: Contents of the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is invalid JSON.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
