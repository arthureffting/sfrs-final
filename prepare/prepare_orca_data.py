import os
import random
from pathlib import Path

import cv2

from prepare.domain.training_image import TrainingImage
from utils.files import read_json, save_to_json
from utils.line_augmentations import LineAugmentation
from utils.line_dewarper import Dewarper2
from utils.painter import Painter

orca_data_folder = os.path.join("data", "orcas")
pages_folder = os.path.join(orca_data_folder, "pages")

year_map = {}

for filename in os.listdir(pages_folder):
    json_path = os.path.join(pages_folder, filename)
    if os.path.isfile(json_path) and filename.endswith("json"):
        year_map[Path(json_path).stem] = read_json(json_path)

all_images = []

for year in year_map:
    data = year_map[year]
    for image_data in data:
        filename = os.path.join(pages_folder, year, image_data["name"])
        if not os.path.exists(filename):
            continue
        image_filename = image_data["name"]
        training_image = TrainingImage({
            "index": image_data["name"],
            "filename": filename,
            "lines": [{
                "index": index,
                "text": "",
                "steps": [{
                    "upper_point": [step["upperPoint"]["x"], step["upperPoint"]["y"]],
                    "base_point": [step["position"]["x"], step["position"]["y"]],
                    "lower_point": [step["lowerPoint"]["x"], step["lowerPoint"]["y"]],
                } for step in line["steps"]]
            } for index, line in enumerate(image_data["lines"])]
        })
        all_images.append(training_image)

print(str(len(all_images)), "images found")

total_lines = 0
for image in all_images:
    total_lines += len(image.lines)

print(str(total_lines), "total_lines found")

split = [0.7, 0.2, 0.1]

training_lines = []
testing_lines = []
evaluation_lines = []

line_index = 0

all_lines = []

for image in all_images:
    if not os.path.exists(image.path):
        continue
    all_lines += image.lines

random.shuffle(all_lines)

for line in all_lines:
    if line_index < split[0] * total_lines:
        training_lines.append(line)
    elif line_index < (split[1] * total_lines + split[0] * total_lines):
        testing_lines.append(line)
    else:
        evaluation_lines.append(line)
    line_index += 1

sets = [training_lines, testing_lines, evaluation_lines]

for index, set in enumerate(["training", "testing", "validation"]):
    # Differently from other datasets, each line will have an entry in the json sets
    set_json_data = []
    set_folder_path = os.path.join("data", "orcas", "prepared", "pages", set)

    for line in sets[index]:
        # Extract line to folder,

        LineAugmentation.normalize(line)
        LineAugmentation.extend_backwards(line)
        LineAugmentation.extend(line, by=6, size_decay=0.9, confidence_decay=0.825)
        LineAugmentation.enforce_minimum_height(line, minimum_height=22)
        LineAugmentation.prevent_wrong_start(line, angle_threshold=30.0)
        line_filename = os.path.join(set_folder_path, line.image.index + "-" + str(line.index) + ".png")
        dewarped_line = Dewarper2(line.steps).dewarped(line.masked())
        cv2.imwrite(line_filename, dewarped_line)
        line_json = [{
            "gt": line.text,
            "image_path": line_filename,
            "steps": [{
                "upper_point": [step.upper_point.x, step.upper_point.y],
                "base_point": [step.base_point.x, step.base_point.y],
                "lower_point": [step.lower_point.x, step.lower_point.y],
                "stop_confidence": step.stop_confidence,
            } for step in line.steps],
            "sol": {
                "x0": line.steps[1].upper_point.x,
                "x1": line.steps[1].lower_point.x,
                "y0": line.steps[1].lower_point.y,
                "y1": line.steps[1].lower_point.y,
            }
        }]

        line_json_path = os.path.join(set_folder_path, line.image.index + "-" + str(line.index) + ".json")
        save_to_json(line_json, line_json_path)
        set_json_data.append([line_json_path, line.image.path])

    json_path = os.path.join("data", "orcas", "prepared", "pages", set + ".json")
    save_to_json(set_json_data, json_path)
