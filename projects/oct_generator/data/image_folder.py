import os
from pathlib import Path

# sadece ihtiyacın olan formatlar
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    """
    Recursively collects image file paths from a directory.

    Args:
        dir (str): root directory
        max_dataset_size (int): limit dataset size

    Returns:
        List[str]: list of image file paths
    """
    images = []
    dir_path = Path(dir)

    if not dir_path.is_dir():
        raise ValueError(f"{dir} is not a valid directory")

    for path in sorted(dir_path.rglob("*")):
        if path.is_file() and is_image_file(path.name):
            images.append(str(path))

    return images[: min(len(images), max_dataset_size)]