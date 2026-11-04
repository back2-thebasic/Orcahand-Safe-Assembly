"""Keep saved MuJoCo scenes portable when comparing or copying runs."""
import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def rebase_scene(source, destination=None):
    source = Path(source).resolve()
    tree = ET.parse(source)
    for element in tree.iter():
        for key, value in list(element.attrib.items()):
            if key.startswith('file'):
                asset = (source.parent / value).resolve()
                element.set(key, str(asset) if destination is None else
                            os.path.relpath(asset, Path(destination).resolve().parent))
    return tree


def scene_signature(path):
    return hashlib.sha256(ET.tostring(rebase_scene(path).getroot())).digest()


def copy_scene(source, destination):
    rebase_scene(source, destination).write(destination, encoding='unicode')
