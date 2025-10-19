import os
import math
import json
from typing import Dict, List, Optional
from torch.utils.data import ConcatDataset

from dataset.doodle_dataset import DoodleDataset
from dataset import doodle_transforms as d
import xml.etree.ElementTree as ET


def split_dataset(default_proportions: Dict[str, float], file_name: str, segmentation_dir: str,
                  class_overrides: Optional[Dict[str, Dict[str, float]]] = None) -> None:

    if class_overrides is None:
        class_overrides = {}

    split_file_path = file_name
    dataset_splits = {key: [] for key in default_proportions.keys()}

    for subfolder in os.listdir(segmentation_dir):
        subfolder_path = os.path.join(segmentation_dir, subfolder)
        if os.path.isdir(subfolder_path):
            json_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder_path) if f.endswith('.json')]
            proportions = class_overrides.get(subfolder, default_proportions)

            start_index = 0
            for i, (key, percentage) in enumerate(proportions.items()):
                split_index = len(json_files) if i == len(proportions) - 1 else start_index + math.ceil(len(json_files) * percentage)
                dataset_splits[key].extend(json_files[start_index:split_index])
                start_index = split_index

    output_dir = os.path.dirname(split_file_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(split_file_path, 'w') as outfile:
        json.dump(dataset_splits, outfile, indent=4)


def split_dataset_xml(default_proportions: Dict[str, float], file_name: str, dataset_dir: str,
                      class_overrides: Optional[Dict[str, Dict[str, float]]] = None) -> None:

    if class_overrides is None:
        class_overrides = {}

    split_file_path = file_name
    dataset_splits = {key: [] for key in default_proportions.keys()}

    for subfolder in os.listdir(dataset_dir):
        subfolder_path = os.path.join(dataset_dir, subfolder)
        proportions = class_overrides.get(subfolder, default_proportions)

        if os.path.isdir(subfolder_path):
            for annotations_folder in os.listdir(subfolder_path):
                annotations_path = os.path.join(subfolder_path, annotations_folder)
                if os.path.isdir(annotations_path):
                    xml_files = [os.path.join(annotations_path, f) for f in os.listdir(annotations_path) if f.endswith('.xml')]
                    all_ids = []

                    for file in xml_files:
                        ids = []
                        tree = ET.parse(file)
                        root = tree.getroot()
                        for element in root.findall('image'):
                            if element.findall('polyline'):
                                ids.append(element.get('id'))
                        all_ids.append(ids)

                    for i, ids in enumerate(all_ids):
                        start_index = 0
                        for k, (key, percentage) in enumerate(proportions.items()):
                            split_index = len(ids) if k == len(proportions) - 1 else start_index + math.ceil(len(ids) * percentage)

                            for j in range(start_index, split_index):
                                entry = {
                                    'file_name': xml_files[i],
                                    'class': subfolder,
                                    'id': ids[j]
                                }
                                dataset_splits[key].append(entry)
                            start_index = split_index

    output_dir = os.path.dirname(split_file_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(split_file_path, 'w') as split_file:
        json.dump(dataset_splits, split_file, indent=4)



def increase_dataset(opt, mode: str, transforms: d.TransformsCompose, number_of_repetitions: int) -> ConcatDataset:

    datasets = []
    for i in range(number_of_repetitions):
        datasets.append(DoodleDataset(data_dir=opt['dataset_path'], split_file_name=opt['data_split_json'], mode=mode, transform=transforms))

    combined_dataset = ConcatDataset(datasets)
    return combined_dataset
