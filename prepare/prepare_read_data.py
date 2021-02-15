import os
from pathlib import Path

from prepare.read_converter import ReadConverter

database_original_folder = os.path.join("data", "read", "original")
target_folder = os.path.join("data", "read", "prepared")

for folder in os.listdir(database_original_folder):
    folder_json_target_path = os.path.join(target_folder, folder + ".json")
    folder_path = os.path.join(database_original_folder, folder)
    pairs = []
    dataset_json_data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".png"):
            stem = Path(file_path).stem
            image_path = os.path.join(folder_path, stem + ".png")
            xml_path = os.path.join(folder_path, "page", stem + ".xml")
            pairs.append([image_path, xml_path])
    for pair in pairs:
        conversion = ReadConverter.convert(pair)

