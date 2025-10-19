import copy
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T

from dataset import doodle_transforms as d
from dataset.doodle_transforms import *
from utils.oop import SingletonMeta


class DoodleObject:

    def __init__(self, strokes: List[List[Tuple[int, int]]], cls_: str, ordered: bool = True,
                 segments: List[List[str]] = None):
        self.strokes = strokes
        self.ordered = ordered
        self.cls = cls_
        self.segments = segments
        self._initialize_additional_properties()

    def _initialize_additional_properties(self):
        self.initial_strokes = None
        self.detector_strokes = None
        self.stroke_point_number = None
        self.rendered_strokes = None
        self.rendered_stroke_locations = None
        self.seg_label = None
        self.seg_label_extend = None
        self.stroke_number = None
        self.key_id = None
        self.category = None

        self.stroke_classes = None  ###

    def __str__(self):
        strokes_count = len(self.strokes)
        ordered_status = "Ordered" if self.ordered else "Unordered"
        strokes_info = ', '.join([str(stroke) for stroke in self.strokes])

        return (f"DoodleObject(class='{self.cls}', "
                f"Strokes Count={strokes_count}, "
                f"Status={ordered_status}, "
                f"Strokes={strokes_info})")


def render_doodle(doodle, offset=(0, 0), scale=1.0):
    transformed_strokes = []
    for stroke in doodle.strokes:
        transformed = [((x * scale + offset[0]), (y * scale + offset[1])) for x, y in stroke]
        transformed_strokes.append(transformed)
    return transformed_strokes


def get_bbox_from_strokes(strokes):
    points = [pt for stroke in strokes for pt in stroke]
    x_coords, y_coords = zip(*points)
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


def strokes_to_polygon(strokes):
    return [coord for stroke in strokes for point in stroke for coord in point]


def create_training_sample(doodles, final_size=800, grid_size=None, padding=10, final_padding=10,
                           scale_jitter=(0.6, 1.2),
                           max_objects=5, save_path=None, sun_cls=3, cloud_cls=1):
    num_doodles = len(doodles)
    # --- Step 1: Filter doodles ---
    # Allow only 1 doodle with sun_cls
    seen_sun = False
    filtered_doodles = []
    for d in doodles:
        if d.cls == sun_cls:
            if not seen_sun:
                filtered_doodles.append(d)
                seen_sun = True
        else:
            filtered_doodles.append(d)

    doodles = filtered_doodles
    num_doodles = len(doodles)

    grid_dim = grid_size or ceil(num_doodles ** 0.5)
    cell_size = final_size // grid_dim

    # --- Step 2: Prioritize placement ---
    # Separate into priority (sun + cloud) and normal
    priority = [d for d in doodles if d.cls in (sun_cls, cloud_cls)]
    normal = [d for d in doodles if d.cls not in (sun_cls, cloud_cls)]

    # Available grid cells
    grid_cells = [(r, c) for r in range(grid_dim) for c in range(grid_dim)]
    random.shuffle(grid_cells)
    placed_objects = []
    placed_objects = []
    used_rows = set()

    for doodle in priority:
        cell = min(grid_cells, key=lambda x: (x[0], x[1]))
        grid_cells.remove(cell)
        placed_objects.append((doodle, cell))
        used_rows.add(cell[0])

    grid_cells = [c for c in grid_cells if c[0] not in used_rows]

    for doodle, cell in zip(normal, grid_cells[:len(normal)]):
        placed_objects.append((doodle, cell))

    objs_out = []
    for doodle, (row, col) in placed_objects:
        cell_origin_x = col * cell_size
        cell_origin_y = row * cell_size

        base_size = int(cell_size * random.uniform(*scale_jitter))
        norm_strokes = rescale_sketch(doodle.strokes, size=base_size, padding=padding)

        bbox = get_bbox(norm_strokes)
        sketch_width, sketch_height = bbox.width(), bbox.height()

        dx = random.randint(0, max(0, cell_size - sketch_width))
        dy = random.randint(0, max(0, cell_size - sketch_height))

        offset = (
            cell_origin_x + dx - bbox.x1,
            cell_origin_y + dy - bbox.y1
        )

        placed_strokes = [
            [(x + offset[0], y + offset[1]) for x, y in line]
            for line in norm_strokes
        ]

        objs_out.append({
            "doodle": doodle,
            "strokes": placed_strokes,
            "cls": doodle.cls
        })

    all_points = [pt for obj in objs_out for stroke in obj["strokes"] for pt in stroke]
    x_coords, y_coords = zip(*all_points)

    combined_x_min, combined_y_min = min(x_coords), min(y_coords)
    combined_x_max, combined_y_max = max(x_coords), max(y_coords)

    total_width = combined_x_max - combined_x_min
    total_height = combined_y_max - combined_y_min

    available_size = final_size - 2 * final_padding
    scale = min(available_size / total_width, available_size / total_height)

    offset_x = (final_size - total_width * scale) / 2
    offset_y = (final_size - total_height * scale) / 2

    img = Image.new("RGB", (final_size, final_size), "white")
    draw = ImageDraw.Draw(img)
    yolo_labels = []

    for obj in objs_out:
        doodle = obj["doodle"]
        scaled_strokes = [
            [((x - combined_x_min) * scale + offset_x,
              (y - combined_y_min) * scale + offset_y)
             for x, y in stroke]
            for stroke in obj["strokes"]
        ]

        doodle.detector_strokes = scaled_strokes
        for stroke in scaled_strokes:
            draw.line(stroke, fill="black", width=2)

        bbox = get_bbox(scaled_strokes)
        x_center = (bbox.x1 + bbox.x2) / 2 / final_size
        y_center = (bbox.y1 + bbox.y2) / 2 / final_size
        width = bbox.width() / final_size
        height = bbox.height() / final_size

        yolo_labels.append([obj["cls"], x_center, y_center, width, height])

    if save_path:
        img.save(save_path)

    img_tensor = T.ToTensor()(img)

    padded_labels = torch.zeros((max_objects, 5), dtype=torch.float32)
    if yolo_labels:
        labels_tensor = torch.tensor(yolo_labels[:max_objects], dtype=torch.float32)
        padded_labels[:labels_tensor.shape[0]] = labels_tensor

    placed_doodles = [obj["doodle"] for obj in objs_out]
    return img_tensor, padded_labels, placed_doodles


class DoodleDatasetDetectorMeta(metaclass=SingletonMeta):

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
            'back-paw',
        ]

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

    def __init__(self, transform: TransformsCompose = None):
        self.transform = transform
        self.meta = DoodleDatasetDetectorMeta()

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
        try:
            sample = self.get_a_sample_detector(random.randint(1, 5))
        except Exception as e:
            print(f"Error at index {idx}")
            raise e

        return sample

    def oversample_paths_by_class(self, all_paths, class_multipliers, get_class_fn):
        oversampled_paths = []
        for path in all_paths:
            cls_ = get_class_fn(path)
            multiplier = class_multipliers.get(cls_, 1)
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

    def _create_label_tensor(self, labels: List[int]) -> torch.Tensor:
        tensor = torch.full((self.meta.MAX_STROKE_NUM,), fill_value=self.meta.EMPTY_INDEX)
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

    def get_sample_from_dict(self, data: Dict, category: int, key_id: int):  # -> Dict[str, Any]:

        doodle = DoodleObject(
            strokes=data["lines"],
            cls_=category,
            segments=self.normalize_segment_labels(data["segments"]),
            ordered=True
        )

        doodle = self._apply_transform(doodle)

        stroke_num = len(doodle.strokes)
        seg_labels_extend, seg_label = self._get_seg_label(doodle)
        strokes = plot_strokes_full(doodle.strokes, self.meta.IMAGE_SIZE)
        strokes = normalize_strokes(strokes, self.meta.IMAGE_SIZE, self.meta.MAX_STROKE_NUM)
        stroke_locations = plot_strokes(doodle.strokes, self.meta.IMAGE_SIZE)
        stroke_locations = normalize_strokes(stroke_locations, self.meta.IMAGE_SIZE, self.meta.MAX_STROKE_NUM)
        initial_strokes, stroke_point_number = self._get_original_strokes(doodle.strokes)  ### [], []

        doodle.seg_label_extend = seg_labels_extend
        doodle.seg_label = seg_label
        doodle.rendered_strokes = strokes
        doodle.rendered_stroke_locations = stroke_locations
        doodle.initial_strokes = initial_strokes
        doodle.stroke_point_number = stroke_point_number
        doodle.stroke_number = stroke_num
        doodle.stroke_classes = [doodle.cls] * stroke_num

        return doodle

    def combine_valid_segmentation_fields(self, doodles: List[DoodleObject]) -> Dict[str, torch.Tensor]:

        all_initial_strokes = []
        all_detector_strokes = []
        all_stroke_point_numbers = []
        all_rendered_strokes = []
        all_rendered_stroke_locations = []
        all_seg_labels = []
        all_seg_labels_extend = []
        all_stroke_classes = []
        all_stroke_indexes = []

        PAD_VALUE = [0, 0]
        MAX_POINTS = self.meta.MAX_POINTS

        for i, doodle in enumerate(doodles):

            stroke_num = doodle.stroke_number
            all_initial_strokes.append(doodle.initial_strokes[:stroke_num])
            detector_strokes = copy.deepcopy(doodle.detector_strokes)

            padded_strokes = []
            for stroke in detector_strokes:
                if len(stroke) > MAX_POINTS:
                    print(f"Stroke too long! Truncating. Class: {doodle.cls}")
                padded_stroke = stroke[:MAX_POINTS] + [PAD_VALUE] * (MAX_POINTS - len(stroke))
                padded_strokes.append(padded_stroke)

            all_detector_strokes.append(torch.tensor(padded_strokes, dtype=torch.float32))
            all_stroke_point_numbers.append(doodle.stroke_point_number[:stroke_num])
            all_rendered_strokes.append(doodle.rendered_strokes[:stroke_num])
            all_rendered_stroke_locations.append(doodle.rendered_stroke_locations[:stroke_num])
            all_seg_labels.append(doodle.seg_label[:stroke_num])
            all_seg_labels_extend.append(doodle.seg_label_extend[:stroke_num])

            all_stroke_classes.extend(doodle.stroke_classes)
            all_stroke_indexes.extend([i] * stroke_num)

        combined = {
            'initial_strokes': torch.cat(all_initial_strokes, dim=0),
            'detector_strokes': torch.cat(all_detector_strokes, dim=0),
            'stroke_point_number': torch.cat(all_stroke_point_numbers, dim=0),
            'strokes': torch.cat(all_rendered_strokes, dim=0),
            'stroke_locations': torch.cat(all_rendered_stroke_locations, dim=0),
            'seg_label': torch.cat(all_seg_labels, dim=0),
            'seg_label_extend': torch.cat(all_seg_labels_extend, dim=0),
            'total_strokes': sum(len(d.strokes) for d in doodles),

            'stroke_classes': torch.tensor(all_stroke_classes),
            'stroke_object_indexes': torch.tensor(all_stroke_indexes)
        }

        return combined

    def get_a_sample_detector(self, num_doodles: int = 5, save_path: str = None) -> Dict[str, Any]:

        selected_paths = random.sample(self.all_paths, num_doodles)
        doodles = []

        for i, path in enumerate(selected_paths):
            try:
                doodle = self.get_sample_from_path(path, key_id=i)
                doodles.append(doodle)
            except Exception as e:
                print(f"Error loading doodle {i} from path {path}: {e}")
                continue

        img, labels, doodles = create_training_sample(doodles=doodles, save_path=save_path)
        combined_seg_data = self.combine_valid_segmentation_fields(doodles)

        return {
            "image": img,
            "labels": labels,
            "num_objects": torch.tensor(num_doodles),
            "initial_strokes": combined_seg_data['initial_strokes'],
            "detector_strokes": combined_seg_data['detector_strokes'],
            "stroke_point_number": combined_seg_data['stroke_point_number'],
            "strokes": combined_seg_data['strokes'],
            "stroke_locations": combined_seg_data['stroke_locations'],
            "seg_label": combined_seg_data['seg_label'],
            "seg_label_extend": combined_seg_data['seg_label_extend'],
            "stroke_number": torch.tensor(combined_seg_data['total_strokes']),
            "stroke_classes": combined_seg_data['stroke_classes'],
            "stroke_object_indexes": combined_seg_data['stroke_object_indexes'],
        }


class DoodleDatasetJSONDetector(DoodleDataset):

    def __init__(self, data_dir: Union[str, Path], split_file_name: str = None, mode: str = None,
                 transform: TransformsCompose = None):
        super().__init__(transform)

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
    def from_paths(paths: List[Path]) -> 'DoodleDatasetJSONDetector':
        dataset = DoodleDatasetJSONDetector.__new__(DoodleDatasetJSONDetector)
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
            cls_ = path.parts[-2]
            if cls_ in self.meta.CLS2IND:
                class_counter[cls_] += 1

        return class_counter

    def get_class_from_json_path(self, path):
        return path.parts[-2]


class DoodleDatasetXMLDetector(DoodleDataset):

    def __init__(self, data_dir: Union[str, Path], split_file_name: str = None, mode: str = None,
                 transform: TransformsCompose = None):
        super().__init__(transform)

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
    def from_paths(paths: List[Path]) -> 'DoodleDatasetXMLDetector':
        dataset = DoodleDatasetXMLDetector.__new__(DoodleDatasetXMLDetector)
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
            cls_ = path['class']
            if cls_ in self.meta.CLS2IND:
                class_counter[cls_] += 1

        return class_counter

    def get_class_from_xml_entry(self, path_dict):
        return path_dict['class']
