import copy
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import torch

from dataset import doodle_transforms as d
from utils.oop import SingletonMeta
from .doodle_object import DoodleObject
from .doodle_transforms import *



class DoodleDatasetMeta(metaclass=SingletonMeta):

    def __init__(self):
        self.IND2CLS = ["car", "cloud", "flower", "sun", "tree", "4leg", "2leg"]
        self.CLS2IND = {cls: ind for ind, cls in enumerate(self.IND2CLS)}

        self.class_multipliers = {
            "car": 1,
            "cloud": 6,
            "flower": 3,
            "sun": 1,
            "tree": 1,
            "4leg": 1,
            "2leg": 4
        }

        self.IND2SEGMENTLABEL = [
            'wheel',
            'frame',

            'center',
            'ray',

            'leaf',
            'petal',
            'stem',

            'body',
            'head',
            'neck',
            'tail',

            'front-foot',
            'back-foot',
            'front-paw',
            'back-paw',

            'steam',
            'leaves',

            'unlabeled-stroke'
        ]

        self.IND2SEGMENTLABEL_EXTEND = [
            'wheel',
            'frame',

            'center',
            'ray',

            'leaf',
            'petal',
            'stem',

            'body',
            'head',
            'neck',
            'tail',

            'front-right-foot',
            'front-left-foot',
            'front-right-paw',
            'front-left-paw',
            'back-right-foot',
            'back-left-foot',
            'back-right-paw',
            'back-left-paw',

            'steam',
            'leaves',

            'unlabeled-stroke'
        ]

        self.general_mapping = {
            'eye': 'head',
            'mouth': 'head',
            'nose': 'head',
            'ear': 'head',
            "eye's apple": 'head',
            'whiskers': 'head',

            'cloud': 'body',
            'trunk': 'steam',
            'crown': 'leaves',
            'leave': 'leaves',
            'branches': 'leaves',
            'prow': 'head'
        }

        self.leg_mapping = {
            'front-right-foot': 'front-foot',
            'front-left-foot': 'front-foot',
            'front-left-paw': 'front-paw',
            'front-right-paw': 'front-paw',
            'back-right-foot': 'back-foot',
            'back-left-foot': 'back-foot',
            'back-right-paw': 'back-paw',
            'back-left-paw': 'back-paw',
        }

        self.classes_w_legs = ["2leg", "4leg"]
        self.stroke_classes_for_clustering = [
            'front-foot',
            'back-foot',
            'front-paw',
            'back-paw', ]

        self.labels_front = [
            'front-right-foot',
            'front-left-foot',
            'front-left-paw',
            'front-right-paw']

        self.labels_back = [
            'back-right-foot',
            'back-left-foot',
            'back-right-paw',
            'back-left-paw']

        self.SEGMENTLABEL2IND = {label: ind for ind, label in enumerate(self.IND2SEGMENTLABEL)}

        self.NUM_STROKE_LABELS = len(self.IND2SEGMENTLABEL)
        self.NUM_SKETCH_CLASSES = len(self.CLS2IND)
        self.EMPTY_INDEX = 42

        self.IMAGE_SIZE = 32
        self.CLASSIFICATOR_IMAGE_SIZE = 64
        self.CLUSTERING_IMAGE_SIZE = 32
        self.MAX_STROKE_NUM = 105
        self.MAX_POINTS = 1000

        self.SEGMENTLABEL2IND_EXTEND = {label: ind for ind, label in enumerate(self.IND2SEGMENTLABEL_EXTEND)}
        self.NUM_STROKE_LABELS_EXTEND = len(self.IND2SEGMENTLABEL_EXTEND)


class DoodleDataset(torch.utils.data.Dataset):

    def __init__(self, transform: TransformsCompose = None, model_type: str = 'segmentator'):
        self.transform = transform
        self.model_type = model_type
        self.meta = DoodleDatasetMeta()

    def num_classes(self) -> int:
        return len(self.meta.IND2CLS)

    def num_clusters(self) -> int:
        return len(self.meta.IND2SEGMENTLABEL)

    def empty_index(self) -> int:
        return self.meta.IND2SEGMENTLABEL.index("<empty>")

    def segmentation_labels(self) -> list:
        return self.meta.IND2SEGMENTLABEL

    def classification_labels(self) -> list:
        return self.meta.IND2CLS

    def __len__(self) -> int:
        return len(self.all_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path: Path = self.all_paths[idx]
        try:
            sample = self.get_sample_from_path(path, key_id=idx)
            if sample is None or (isinstance(sample, dict) and not sample):
                print(f"[WARN] Empty sample at idx {idx}, path={path}")
            return sample
        except Exception as e:
            print(f"[ERROR] Failed to load sample at idx {idx}, path={path}, error={e}")
            return None

    def oversample_paths_by_class(self, all_paths, class_multipliers, get_class_fn):

        oversampled_paths = []
        for path in all_paths:
            cls = get_class_fn(path)
            multiplier = class_multipliers.get(cls, 1)
            oversampled_paths.extend([path] * multiplier)

        return oversampled_paths

    def normalize_segment_labels(self, segments):
        result = []
        for segment in segments:
            label_counts = Counter(segment)
            most_common_label = label_counts.most_common(1)[0][0]
            result.append(most_common_label)
        return result

    def _get_seg_label(self, data: DoodleObject) -> Tuple[torch.Tensor, torch.Tensor]:

        int_labels, int_labels_leg_extend = [], []
        for segment in data.segments:

            leg_label_extend = copy.deepcopy(segment)
            if leg_label_extend in self.meta.general_mapping:
                leg_label_extend = self.meta.general_mapping[leg_label_extend]

            label = copy.deepcopy(leg_label_extend)
            if label in self.meta.leg_mapping:
                label = self.meta.leg_mapping[label]

            try:
                int_labels.append(self.meta.SEGMENTLABEL2IND[label])
                int_labels_leg_extend.append(self.meta.SEGMENTLABEL2IND_EXTEND[leg_label_extend])

            except KeyError as e:
                print(f"KeyError: {e} in segment {segment}")
                raise

        return (
            self._create_label_tensor(int_labels_leg_extend),
            self._create_label_tensor(int_labels)
        )

    def _create_label_tensor(self, labels: List[int], empty_index: int = None) -> torch.Tensor:

        if empty_index is None:
            empty_index = self.meta.EMPTY_INDEX
        tensor = torch.full((self.meta.MAX_STROKE_NUM,), fill_value=empty_index)
        tensor[:len(labels)] = torch.tensor(labels)
        return tensor

    def _get_original_strokes(self, original_strokes) -> Tuple[torch.Tensor, torch.Tensor]:

        padded_strokes = copy.deepcopy(original_strokes)
        PAD_VALUE = [0, 0]
        stroke_point_number = []
        for i in range(len(padded_strokes)):

            stroke_point_number.append(len(padded_strokes[i]))
            if len(padded_strokes[i]) > self.meta.MAX_POINTS:
                print('Stroke too long!!!')
            padded_strokes[i] += [PAD_VALUE] * (self.meta.MAX_POINTS - len(padded_strokes[i]))

        empty_stroke = [[0, 0]] * self.meta.MAX_POINTS
        padded_strokes += [empty_stroke] * (self.meta.MAX_STROKE_NUM - len(padded_strokes))
        padded_strokes = torch.tensor(padded_strokes, dtype=torch.float32)

        stroke_point_number += [0] * (self.meta.MAX_STROKE_NUM - len(stroke_point_number))
        stroke_point_number = torch.tensor(stroke_point_number)

        return padded_strokes, stroke_point_number

    def _apply_transform(self, doodle: DoodleObject) -> DoodleObject:

        if not self.transform:
            return doodle

        original = copy.deepcopy(doodle)
        try:
            return self.transform(doodle)
        except Exception as e:
            print(f"Augmentation error: {e}")
            return original

    def get_sample_from_dict(self, data: Dict, category: int, key_id: int) -> Dict[str, Any]:

        doodle = DoodleObject(
            strokes=data["lines"],
            cls=category,
            segments=self.normalize_segment_labels(data["segments"]),
            ordered=True
        )

        if self.model_type == 'clustering':
            return self._create_clustering_sample(doodle, key_id)

        else:
            doodle = self._apply_transform(doodle)
            return self._create_segmentation_sample(doodle, key_id)

    def get_leg_label_index(self, label: str) -> int:
        if 'right' in label:
            return 0
        elif 'left' in label:
            return 1
        raise ValueError(f"Invalid leg orientation label: {label}")

    def split_doodle_by_leg(self, doodle: DoodleObject, seg_labels: List[int]) -> Tuple[DoodleObject, DoodleObject]:

        front_strokes, front_segments = [], []
        back_strokes, back_segments = [], []

        for i, stroke in enumerate(doodle.strokes):
            seg_idx = seg_labels[i]
            label = self.meta.IND2SEGMENTLABEL_EXTEND[seg_idx]

            if label in self.meta.labels_front:
                front_strokes.append(stroke)
                front_segments.append(label)
            elif label in self.meta.labels_back:
                back_strokes.append(stroke)
                back_segments.append(label)

        return (
            DoodleObject(strokes=front_strokes, cls=doodle.cls, segments=front_segments, ordered=True),
            DoodleObject(strokes=back_strokes, cls=doodle.cls, segments=back_segments, ordered=True)
        )

    def create_one_leg_sample(self, doodle: DoodleObject, key_id: int) -> Dict[str, Any]:

        doodle = self._apply_transform(doodle)
        strokes = plot_strokes(doodle.strokes, self.meta.CLUSTERING_IMAGE_SIZE)
        strokes = normalize_strokes(strokes, self.meta.CLUSTERING_IMAGE_SIZE, self.meta.MAX_STROKE_NUM)

        label_legs = []
        label_segments = []
        for label in doodle.segments:
            label_legs.append(self.get_leg_label_index(label))
            label_segments.append(self.meta.stroke_classes_for_clustering.index(self.meta.leg_mapping[label]))

        seg_label = self._create_label_tensor(label_legs, empty_index=4)
        stroke_label = self._create_label_tensor(label_segments, empty_index=4)
        category = self.meta.classes_w_legs.index(self.meta.IND2CLS[doodle.cls])

        return {
            'strokes': strokes,
            'category': torch.tensor(category),
            'seg_label': seg_label,
            'stroke_labels': stroke_label,
            'stroke_number': torch.tensor(len(doodle.strokes)),
            'key_id': key_id
        }

    def _create_clustering_sample(self, doodle: DoodleObject, key_id: int) -> List[Dict]:
        seg_labels_extend, _ = self._get_seg_label(doodle)
        front, back = self.split_doodle_by_leg(doodle, seg_labels_extend)
        return [
            self.create_one_leg_sample(front, key_id),
            self.create_one_leg_sample(back, key_id)
        ]

    def _create_segmentation_sample(self, doodle: DoodleObject, key_id: int) -> Dict:

        seg_labels_extend, seg_label = self._get_seg_label(doodle)
        strokes = plot_strokes(doodle.strokes, self.meta.IMAGE_SIZE)
        strokes = normalize_strokes(strokes, self.meta.IMAGE_SIZE, self.meta.MAX_STROKE_NUM)
        strokes_full = plot_strokes_full(doodle.strokes, self.meta.IMAGE_SIZE)
        strokes_full = normalize_strokes(strokes_full, self.meta.IMAGE_SIZE, self.meta.MAX_STROKE_NUM)
        initial_strokes, stroke_point_number = self._get_original_strokes(doodle.strokes)  ### [], []
        category = doodle.cls

        return {
            'initial_strokes': initial_strokes,
            'stroke_point_number': stroke_point_number,
            'strokes': strokes_full,
            'stroke_locations': strokes,
            'category': torch.tensor(category),
            'seg_label': seg_label,
            'seg_label_extend': seg_labels_extend,
            'stroke_number': torch.tensor(len(doodle.strokes)),
            'key_id': key_id,
        }


class DoodleDatasetJSON(DoodleDataset):

    def __init__(self, data_dir: Union[str, Path], split_file_name: str = None, mode: str = None,
                 transform: TransformsCompose = None, model_type: str = 'segmentator'):
        super().__init__(transform, model_type)

        if model_type == 'clustering':
            classes = self.meta.classes_w_legs
        else:
            classes = self.meta.IND2CLS

        self.data_dir = Path(data_dir)
        self.mode = mode

        if mode is not None and split_file_name is not None:
            split_file_path = split_file_name  # self.data_dir / split_file_name
            with open(split_file_path, 'r') as f:
                split_data = json.load(f)

            if mode in split_data:
                paths = []
                for path in split_data[mode]:
                    dir = path.split('/')[1] if path.startswith('/') else path.split('/')[0]
                    if dir in classes:
                        paths.append(self.data_dir / path)

                if mode == 'train':
                    self.all_paths = self.oversample_paths_by_class(paths, self.meta.class_multipliers,
                                                                    self.get_class_from_json_path)
                else:
                    self.all_paths = paths

            else:
                raise ValueError(f"Mode '{mode}' not found in the split file.")

        else:
            self.all_paths = list(self.data_dir.glob(f"*/*.json"))

    @staticmethod
    def from_paths(paths: List[Path]) -> 'DoodleDatasetJSON':
        dataset = DoodleDatasetJSON.__new__(DoodleDatasetJSON)
        dataset.all_paths = paths
        return dataset

    def get_sample_from_path(self, path: Path, category: int = None, key_id: int = -1) -> Dict[str, Any]:
        data = self._load_data(path)
        category = category or self.meta.CLS2IND[path.parent.name]
        return self.get_sample_from_dict(data, category=category, key_id=key_id)

    def _load_data(self, path: Path) -> Dict[str, Any]:
        with open(path, 'r') as f:
            data = json.load(f)

        return data

    def get_class_counts(self):
        class_counter = defaultdict(int)
        for path in self.all_paths:
            cls = path.parts[-2]
            if cls in self.meta.CLS2IND:
                class_counter[cls] += 1

        return class_counter

    def get_class_from_json_path(self, path):
        return path.parts[-2]


class DoodleDatasetXML(DoodleDataset):

    def __init__(self, data_dir: Union[str, Path], split_file_name: str = None, mode: str = None,
                 transform: TransformsCompose = None, model_type: str = 'segmentator'):
        super().__init__(transform, model_type)

        if model_type == 'clustering':
            classes = self.meta.classes_w_legs
        else:
            classes = self.meta.IND2CLS

        self.data_dir = Path(data_dir)
        self.mode = mode

        if split_file_name is not None:
            split_file_path = split_file_name  # self.data_dir / split_file_name
            with open(split_file_path, 'r') as f:
                split_data = json.load(f)

            if mode in split_data:
                paths = []
                for path in split_data[mode]:
                    if path['class'] in classes and self.right_number_of_strokes(path):
                        paths.append(path)

                if mode == 'train':
                    self.all_paths = self.oversample_paths_by_class(paths, self.meta.class_multipliers,
                                                                    self.get_class_from_xml_entry)
                else:
                    self.all_paths = paths

            else:
                raise ValueError(f"Mode '{mode}' not found in the split file.")

        else:
            print("No split file!")

    @staticmethod
    def from_paths(paths: List[Path]) -> 'DoodleDatasetXML':
        dataset = DoodleDatasetXML.__new__(DoodleDatasetXML)
        dataset.all_paths = paths
        return dataset

    def right_number_of_strokes(self, path) -> bool:
        file_path = path['file_name']
        id = path['id']
        data = self._load_data(file_path, id)
        if len(data['lines']) <= self.meta.MAX_STROKE_NUM:
            return True
        return False

    def get_sample_from_path(self, path: Dict[str, Any], category: int = None, key_id: int = -1) -> Dict[str, Any]:
        file_path = path['file_name']
        doodle_class = path['class']
        id = path['id']
        data = self._load_data(file_path, id)
        category = category or self.meta.CLS2IND[doodle_class]
        return self.get_sample_from_dict(data, category=category, key_id=key_id)

    def _load_data(self, path: str, id: str) -> Dict[str, Any]:
        tree = ET.parse(path)
        root = tree.getroot()

        for element in root.findall('image'):
            if element.get('id') == id:
                polylines = element.findall('polyline')
                if polylines:
                    lines = []
                    segments = []
                    for stroke in polylines:
                        points_raw = stroke.get('points')
                        points = [
                            [float(coord) for coord in point.split(',')]
                            for point in points_raw.split(';')
                        ]

                        segment = [stroke.get('label').lower()]
                        lines.append(points)
                        segments.append(segment)
                    return {"lines": lines, "segments": segments}
        return {}

    def get_class_counts(self):
        class_counter = defaultdict(int)
        for path in self.all_paths:
            cls = path['class']
            if cls in self.meta.CLS2IND:
                class_counter[cls] += 1

        return class_counter

    def get_class_from_xml_entry(self, path_dict):
        return path_dict['class']
