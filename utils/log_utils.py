import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from clearml import Task

from dataset.doodle_transforms import TransformsCompose
from utils.utils import ModelOutputs


class FilePaths:
    def __init__(self, paths_opt):

        self.model_weights_folder = paths_opt['home'] + paths_opt['experiment_folder'] + paths_opt['ckpt_folder']
        self.log_folder = paths_opt['home'] + paths_opt['experiment_folder'] + paths_opt['log_folder']
        self.model_outputs_folder = self.log_folder + paths_opt['model_outputs_folder']
        self.log_file = paths_opt['log_file']
        self.raw_output_file = paths_opt['raw_outputs_file']

    def check_all(self) -> None:
        self.check_folder(self.model_weights_folder)
        self.check_folder(self.log_folder)
        self.check_folder(self.model_outputs_folder)
        self.check_file(self.log_file, self.log_folder)

    def check_folder(self, folder: str) -> None:
        print('folder ' + folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

    def check_file(self, file: str, folder: str) -> None:
        file = folder + file
        print(file)
        if not os.path.exists(file):
            os.mknod(file)


def log_experiment(log):
    if log:
        task = Task.init(project_name='stroke_segmentation', task_name='model_training',
                         reuse_last_task_id=False)
        print("Experiment logging enabled")
    else:
        print("Experiment logging disabled")


def write_training_metrics(epoch_id: int, valid_loss: float, valid_model_outputs: ModelOutputs, trial_number: int,
                           file_paths: FilePaths) -> None:
    the_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_path = os.path.join(file_paths.log_folder, file_paths.log_file)

    log_message = (
        f"Time: {the_time}; Epoch: {epoch_id}; Loss: {valid_loss:.4f}; "
        f"Classification accuracy: {valid_model_outputs.get_classification_accuracy():.4f}; "
        f"Segmentation accuracy: {valid_model_outputs.get_segmentation_accuracy():.4f}\n"
    )
    if trial_number is not None:
        log_message = f'Trial {trial_number}: ' + log_message

    with open(log_path, 'a') as text_file:
        text_file.write(log_message)


def write_raw_outputs(epoch_id: int, valid_model_outputs: ModelOutputs, trial_number: int,
                      file_paths: FilePaths) -> None:
    if trial_number is not None:
        output_path = os.path.join(file_paths.model_outputs_folder,
                                   f'trial_{trial_number}_' + file_paths.raw_output_file)
    else:
        output_path = os.path.join(file_paths.model_outputs_folder, file_paths.raw_output_file)

    output_data = valid_model_outputs.raw_outputs_to_dict(digits=4)
    output_data['Epoch'] = epoch_id
    with open(output_path, 'a') as f:
        json.dump(output_data, f)
        f.write('\n')


def display_experiment_params(model: Any = None, transforms: TransformsCompose = None, train_set: Any = None,
                              dataset_opt: Dict[str, Any] = None) -> None:
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}\n")
        print(f"Model Architecture:\n{model}\n")
    if train_set is not None:
        print(f"Batch size: {dataset_opt['bs']}\n")
        print(f"Total number of train dataset instances: {len(train_set)}\n")
    if transforms is not None:
        print("Augmentations:")
        for transform in transforms.transforms:
            print(f"{transform}")


def save_checkpoint(model: Any, url: str, opt: Dict[str, Any], label_mapping: List = None) -> None:
    model_folder = Path(opt['experiment_folder']) / opt['ckpt_folder']
    model_folder.mkdir(parents=True, exist_ok=True)
    model_path = model_folder / url
    print('Saving checkpoint:', model_path)
    torch.save(model.state_dict(), model_path)

    if label_mapping is not None:
        label_map_path = model_folder / "label_mapping.txt"
        with label_map_path.open("w") as f:
            if all(isinstance(l, list) for l in label_mapping):
                for i, sublist in enumerate(label_mapping):
                    f.write(f"Task {i}:\n")
                    for idx, label in enumerate(sublist):
                        f.write(f"{idx}: {label}\n")
                    f.write("\n")
            else:
                for idx, label in enumerate(label_mapping):
                    f.write(f"{idx}: {label}\n")
        print("Label mapping saved:", label_map_path)
    else:
        print("No label mapping provided. Skipping label mapping save.")


def load_checkpoint(model: Any, path: str) -> None:
    print(f'Loading checkpoint: {path}')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
