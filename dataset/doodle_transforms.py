import abc
import copy
import math
import os
import random
import sys
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from .doodle_object import BBoxInt, DoodleObject


def rotate_point(p: Tuple[int, int], origin: Tuple[int, int] = (0, 0), degrees: float = 0) -> np.array:
    """
    Rotate point around origin by degrees

    Parameters
    ----------
    p : Tuple[int, int]
        Point to rotate (x, y) format

    origin : Tuple[int, int]
        Origin of rotation (x, y) format

    degrees : float
        Angle in degrees

    Returns
    -------
    np.array
        1d array with rotated point (x, y) format

    """

    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def get_bbox(doodle: Any) -> BBoxInt:
    curr_size = [float('inf'), float('inf'), -float('inf'), -float('inf')]  # x_min, y_min, x_max, y_max
    if isinstance(doodle, List):
        strokes = doodle
    else:
        strokes = doodle.strokes

    # print(doodle)
    for stroke in strokes:
        for point in stroke:
            for i in range(2):  # x, y
                curr_size[i] = min(curr_size[i], point[i])
                curr_size[i + 2] = max(curr_size[i + 2], point[i])

    return BBoxInt(curr_size)


def get_stroke_bbox(stroke: List[Tuple[int, int]]) -> BBoxInt:
    curr_size = [float('inf'), float('inf'), -float('inf'), -float('inf')]  # x_min, y_min, x_max, y_max
    for point in stroke:
        for i in range(2):  # x, y
            curr_size[i] = min(curr_size[i], point[i])
            curr_size[i + 2] = max(curr_size[i + 2], point[i])

    return BBoxInt(curr_size)


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def apply_transformation(p: float = 0):
    return random.choices([True, False], weights=[p, 1 - p])[0]


def find_point_on_line(a: Tuple[float, float], b: Tuple[float, float], distance: float, margin: float) -> List[float]:
    x_a, y_a = a
    x_b, y_b = b
    x_c = x_a + margin * (x_b - x_a) / distance
    y_c = y_a + margin * (y_b - y_a) / distance
    new_point = [x_c, y_c]
    return new_point


class BaseTransform(abc.ABC):
    # abc.abstractmethod
    def __call__(self, doodle: DoodleObject) -> DoodleObject: ...


class TransformsCompose(BaseTransform):
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms

    # override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:

        for transform in self.transforms:    
            doodle = transform(doodle)

        return doodle


class Rotate(BaseTransform):
    def __init__(self, angle: float = 0, inplace: bool = False):
        self.angle = angle
        self.inplace = inplace

    # override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not self.inplace:
            doodle = copy.deepcopy(doodle)

        return self.rotate(self.angle, doodle)

    def rotate(self, ang: float, doodle: DoodleObject) -> DoodleObject:

        bbox = get_bbox(doodle)
        w = bbox.width()
        h = bbox.height()

        central_point = np.asarray([w / 2.0, h / 2.0], dtype=np.float32)

        for stroke in doodle.strokes:
            for i in range(0, len(stroke)):
                res_p = rotate_point(stroke[i], central_point, ang)
                stroke[i] = [int(res_p[0]), int(res_p[1])]

        return doodle


class Scale(BaseTransform):
    def __init__(self, fx: float = 1, fy: float = 1, inplace: bool = False):
        self.fx = fx
        self.fy = fy
        self.inplace = inplace

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not self.inplace:
            doodle = copy.deepcopy(doodle)

        return self.scale(doodle, fx=self.fx, fy=self.fy)

    def scale(self, doodle: DoodleObject, fx: float, fy: float):

        ff = [fx, fy]
        for stroke in doodle.strokes:
            for point in stroke:
                for i in range(2):  # x, y
                    point[i] = int(point[i] * ff[i])
        return doodle


class Shift(BaseTransform):
    def __init__(self, dx: float = 0, dy: float = 0, inplace: bool = False):
        self.dx = dx
        self.dy = dy
        self.inplace = inplace

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not self.inplace:
            doodle = copy.deepcopy(doodle)

        return self.shift(doodle, dx=self.dx, dy=self.dy)

    def shift(self, doodle: DoodleObject, dx: float, dy: float) -> DoodleObject:

        for stroke in doodle.strokes:
            for point in stroke:
                point[0] = int(point[0] + dx)
                point[1] = int(point[1] + dy)
        return doodle


class PlaceTo(BaseTransform):

    def __init__(self, dx: int = 0, dy: int = 0, inplace: bool = False):
        self.dx = dx
        self.dy = dy
        self.inplace = inplace

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not self.inplace:
            doodle = copy.deepcopy(doodle)

        return self.place_to(doodle, x=self.dx, y=self.dy)

    def place_to(self, doodle: DoodleObject, x: int, y: int) -> DoodleObject:
        curr_size = get_bbox(doodle)

        dx = x - curr_size.x1
        dy = y - curr_size.y1

        for stroke in doodle.strokes:
            for point in stroke:
                point[0] = int(point[0] + dx)
                point[1] = int(point[1] + dy)

        return doodle


class HorizontalFlip(BaseTransform):
    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not self.inplace:
            doodle = copy.deepcopy(doodle)

        return self.horizontal_flip(doodle)

    def horizontal_flip(self, doodle: DoodleObject) -> DoodleObject:
        bbox = get_bbox(doodle)
        h = bbox.height()

        for stroke in doodle.strokes:
            for i in range(0, len(stroke)):
                stroke[i] = [stroke[i][0], h - stroke[i][1] - 1]
        return doodle


class VerticalFlip(BaseTransform):
    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not self.inplace:
            doodle = copy.deepcopy(doodle)

        return self.vertical_flip(doodle)

    def vertical_flip(self, doodle: DoodleObject) -> DoodleObject:
        bbox = get_bbox(doodle)
        w = bbox.width()

        mapping = {
            "front-right-foot": "front-left-foot",
            "front-left-foot": "front-right-foot",
            "back-right-foot": "back-left-foot",
            "back-left-food": "back-right-foot",
            'front-left-paw': "front-right-paw",
            'front-right-paw': "front-left-paw",
            'back-right-paw': "back-left-paw",
            'back-left-paw': "back-right-paw"
        }

        for i, stroke in enumerate(doodle.strokes):
            for j in range(0, len(stroke)):
                stroke[j] = [w - stroke[j][0] - 1, stroke[j][1]]
            if doodle.segments[i] in mapping:
                doodle.segments[i] = mapping[doodle.segments[i]]
        return doodle


class Resize(BaseTransform):
    def __init__(self, new_bbox: Tuple[int, int] = (100, 100), inplace: bool = False, without_shift: bool = False):
        self.inplace = inplace
        self.without_shift = without_shift
        self.new_bbox = new_bbox

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not self.inplace:
            doodle = copy.deepcopy(doodle)

        if self.without_shift:
            return self.resize_(doodle, self.new_bbox)
        else:
            return self.resize_to_bbox_(doodle, self.new_bbox)

    def resize_to_bbox_(self, doodle: DoodleObject, bbox_size: Union[List[int], Tuple[int, int]]) -> DoodleObject:

        curr_size = get_bbox(doodle)
        start_position = [curr_size.x1, curr_size.y1]
        end_position = [curr_size.x2, curr_size.y2]
        for stroke in doodle.strokes:
            for point in stroke:
                for i in range(2):  # x, y
                    point[i] = int(
                        (point[i] - start_position[i]) / (end_position[i] - start_position[i]) * bbox_size[i]) + \
                               start_position[i]
        return doodle

    def resize_(self, doodle: DoodleObject, size: Tuple[int, int]) -> DoodleObject:

        curr_size = get_bbox(doodle)
        end_position = [curr_size.x2, curr_size.y2]
        for stroke in doodle.strokes:
            for point in stroke:
                for i in range(2):  # x, y
                    point[i] = int(point[i] / end_position[i] * size[i])
        return doodle


class RandomRotate(BaseTransform):
    def __init__(self, angle_1: float, angle_2, p: float = 0.5, inplace: bool = False):
        self.angle_1 = angle_1
        self.angle_2 = angle_2
        self.p = p
        self.inplace = inplace
        self.rotate_instance = Rotate(inplace=self.inplace)

    # override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not apply_transformation(p=self.p):
            return doodle

        angle = random.uniform(self.angle_1, self.angle_2)
        self.rotate_instance.angle = angle
        return self.rotate_instance(doodle)

    def __str__(self) -> str:
        return (
            f"RandomRotate(angle_1={self.angle_1}, angle_2={self.angle_2}, "
            f"p={self.p}, inplace={self.inplace})"
        )


class RandomScaleOR(BaseTransform):
    def __init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float], p: float = 0.5,
                 inplace: bool = False):
        self.x_range = x_range
        self.y_range = y_range
        self.p = p
        self.inplace = inplace
        self.scale_instance = Scale(inplace=self.inplace)

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not apply_transformation(p=self.p):
            return doodle

        if random.choice([True, False]):
            fx = random.uniform(self.x_range[0], self.x_range[1])
            fy = 1.0
        else:
            fx = 1.0
            fy = random.uniform(self.y_range[0], self.y_range[1])

        self.scale_instance.fx = fx
        self.scale_instance.fy = fy
        return self.scale_instance(doodle)

    def __str__(self) -> str:
        return (
            f"RandomScaleOR(x_range={self.x_range}, y_range={self.y_range}, "
            f"p={self.p}, inplace={self.inplace})"
        )


class RandomShift(BaseTransform):
    def __init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float], p: float = 0.5, inplace: bool = False):
        self.x_range = x_range
        self.y_range = y_range
        self.p = p
        self.inplace = inplace
        self.shift_instance = Shift(inplace=self.inplace)

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not apply_transformation(p=self.p):
            return doodle
        dx = random.uniform(self.x_range[0], self.x_range[1])
        dy = random.uniform(self.y_range[0], self.y_range[1])
        self.shift_instance.dx = dx
        self.shift_instance.dy = dy
        return self.shift_instance(doodle)

    def __str__(self) -> str:
        return (
            f"RandomShift(x_range={self.x_range}, y_range={self.y_range}, "
            f"p={self.p}, inplace={self.inplace})"
        )


class RandomVerticalFlip(BaseTransform):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        self.p = p
        self.inplace = inplace
        self.flip_instance = VerticalFlip(inplace=self.inplace)

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not apply_transformation(p=self.p):
            return doodle
        if self.inplace:
            doodle = copy.deepcopy(doodle)

        if random.choice([True, False]):
            return self.flip_instance(doodle)
        else:
            return doodle

    def __str__(self) -> str:
        return f"RandomVerticalFlip(p={self.p}, inplace={self.inplace})"


class RandomResize(BaseTransform):
    def __init__(self, x_range: Tuple[int, int], y_range: Tuple[int, int], p: float = 0.5, inplace: bool = False,
                 without_shift: bool = False):
        self.x_range = x_range
        self.y_range = y_range
        self.p = p
        self.inplace = inplace
        self.without_shift = without_shift
        self.resize_instance = Resize(inplace=self.inplace, without_shift=self.without_shift)

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not apply_transformation(p=self.p):
            return doodle
        new_x = random.randint(self.x_range[0], self.x_range[1])
        new_y = random.randint(self.y_range[0], self.y_range[1])
        self.resize_instance.new_bbox = (new_x, new_y)
        return self.resize_instance(doodle)

    def __str__(self) -> str:
        return (
            f"RandomResize(x_range={self.x_range}, y_range={self.y_range}, "
            f"p={self.p}, inplace={self.inplace}, without_shift={self.without_shift})"
        )


class StrokeDropout(BaseTransform):
    def __init__(self, dropout_num: int = 1, min_sketch_strokes: int = 3, p: float = 0.1, smallest: bool = False,
                 inplace: bool = False):
        self.dropout_num = dropout_num
        self.min_sketch_strokes = min_sketch_strokes
        self.p = p
        self.smallest = smallest
        self.inplace = inplace

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if len(doodle.strokes) - self.dropout_num < self.min_sketch_strokes:
            return doodle
        if self.inplace:
            doodle = copy.deepcopy(doodle)

        if self.smallest:
            sorted_indices = self.sort_by_stroke_length(doodle)
            indices = sorted_indices[:self.dropout_num]
        else:
            total_strokes = len(doodle.strokes)
            indices = random.sample(range(total_strokes), self.dropout_num)

        indices.sort(reverse=True)
        return self.remove_strokes(indices, doodle)

    def sort_by_stroke_length(self, doodle: DoodleObject) -> list:
        stroke_lengths = []
        for stroke in doodle.strokes:
            total_length = sum(euclidean_distance(stroke[i], stroke[i + 1]) for i in range(len(stroke) - 1))
            stroke_lengths.append(total_length)
        sorted_indices = sorted(range(len(stroke_lengths)), key=lambda i: stroke_lengths[i])
        return sorted_indices

    def remove_strokes(self, indices: list, doodle: DoodleObject) -> DoodleObject:

        for index in indices:
            if not apply_transformation(p=self.p):
                continue
            doodle.strokes.pop(index)
            doodle.segments.pop(index)

        return doodle

    def __str__(self) -> str:
        return (
            f"StrokeDropout(dropout_num={self.dropout_num}, min_sketch_strokes={self.min_sketch_strokes}, "
            f"p={self.p}, smallest={self.smallest}, inplace={self.inplace})"
        )


class RandomStrokeRotate(BaseTransform):
    def __init__(self, angle_range: Tuple[float, float], stroke_num: int = 1, p: float = 0.5, inplace: bool = False):
        self.angle_range = angle_range
        self.stroke_num = stroke_num
        self.p = p
        self.inplace = inplace

    # override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not self.inplace:
            doodle = copy.deepcopy(doodle)

        stroke_num = min(self.stroke_num, len(doodle.strokes))
        indices = random.sample(range(len(doodle.strokes)), stroke_num)
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        for index in indices:
            if not apply_transformation(p=self.p):
                continue
            doodle.strokes[index] = self.rotate_stroke(angle, doodle.strokes[index])

        return doodle

    def rotate_stroke(self, ang: float, stroke: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

        bbox = get_stroke_bbox(stroke)
        w = bbox.width()
        h = bbox.height()
        central_point = np.asarray([w / 2.0, h / 2.0], dtype=np.float32)

        for i in range(0, len(stroke)):
            res_p = rotate_point(stroke[i], central_point, ang)
            stroke[i] = [int(res_p[0]), int(res_p[1])]

        return stroke

    def __str__(self) -> str:
        return (
            f"RandomStrokeRotate(angle_range={self.angle_range}, stroke_num={self.stroke_num}, "
            f"p={self.p}, inplace={self.inplace})"
        )


class RandomStrokeScaleOR(BaseTransform):
    def __init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float], stroke_num: int = 1, p: float = 0.5,
                 inplace: bool = False):
        self.x_range = x_range
        self.y_range = y_range
        self.p = p
        self.stroke_num = stroke_num
        self.inplace = inplace

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not self.inplace:
            doodle = copy.deepcopy(doodle)

        stroke_num = min(self.stroke_num, len(doodle.strokes))
        indices = random.sample(range(len(doodle.strokes)), stroke_num)
        if random.choice([True, False]):
            fx = random.uniform(self.x_range[0], self.x_range[1])
            fy = 1.0
        else:
            fx = 1.0
            fy = random.uniform(self.y_range[0], self.y_range[1])

        for index in indices:
            if not apply_transformation(p=self.p):
                continue
            doodle.strokes[index] = self.scale_stroke(fx, fy, doodle.strokes[index])
        return doodle

    def scale_stroke(self, fx: float, fy: float, stroke: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

        ff = [fx, fy]
        for point in stroke:
            for i in range(2):  # x, y
                point[i] = int(point[i] * ff[i])
        return stroke

    def __str__(self) -> str:
        return (
            f"RandomStrokeScaleOR(x_range={self.x_range}, y_range={self.y_range}, "
            f"stroke_num={self.stroke_num}, p={self.p}, inplace={self.inplace})"
        )


class RandomStrokeShift(BaseTransform):
    def __init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float], stroke_num: int = 1, p: float = 0.5,
                 inplace: bool = False):
        self.x_range = x_range
        self.y_range = y_range
        self.p = p
        self.stroke_num = stroke_num
        self.inplace = inplace

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not self.inplace:
            doodle = copy.deepcopy(doodle)

        stroke_num = min(self.stroke_num, len(doodle.strokes))
        indices = random.sample(range(len(doodle.strokes)), stroke_num)
        dx = random.uniform(self.x_range[0], self.x_range[1])
        dy = random.uniform(self.y_range[0], self.y_range[1])
        for index in indices:
            if not apply_transformation(p=self.p):
                continue
            doodle.strokes[index] = self.shift_stroke(dx, dy, doodle.strokes[index])
        return doodle

    def shift_stroke(self, dx: float, dy: float, stroke: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

        for point in stroke:
            point[0] = int(point[0] + dx)
            point[1] = int(point[1] + dy)
        return stroke

    def __str__(self) -> str:
        return (
            f"RandomStrokeShift(x_range={self.x_range}, y_range={self.y_range}, "
            f"stroke_num={self.stroke_num}, p={self.p}, inplace={self.inplace})"
        )


class PointDensityAugmentation(BaseTransform):
    def __init__(self, min_length_fraction: float, inplace: bool = False):
        self.min_length_fraction = min_length_fraction
        self.inplace = inplace

    # @override
    def __call__(self, doodle: DoodleObject) -> DoodleObject:
        if not self.inplace:
            doodle = copy.deepcopy(doodle)

        doodle_bbox = get_bbox(doodle)
        width = doodle_bbox.x2 - doodle_bbox.x1
        height = doodle_bbox.y2 - doodle_bbox.y1
        length_between_points = max(width, height) * self.min_length_fraction

        for index in range(len(doodle.strokes)):
            doodle.strokes[index] = self.adjust_point_density(doodle.strokes[index], length_between_points)
        return doodle

    def adjust_point_density(self, stroke: List[Tuple[int, int]], length_between_points: float) -> List[
        Tuple[int, int]]:

        if len(stroke) <= 3:
            return stroke
        adjusted_stroke = [stroke[0]]
        cumulative_distance = 0
        for i in range(1, len(stroke)):

            last_stroke_point = adjusted_stroke[-1]
            point_a = stroke[i - 1]
            point_b = stroke[i]
            distance = math.dist(last_stroke_point, point_b)
            current_distance = math.dist(point_a, point_b)
            cumulative_distance += current_distance

            while distance >= length_between_points:
                new_point = find_point_on_line(last_stroke_point, point_b, distance, length_between_points)
                adjusted_stroke.append(new_point)
                last_stroke_point = new_point
                distance = math.dist(last_stroke_point, point_b)
                cumulative_distance = 0

            if cumulative_distance >= length_between_points:
                margin = length_between_points + current_distance - cumulative_distance
                new_point = find_point_on_line(point_a, point_b, current_distance, margin)
                adjusted_stroke.append(new_point)
                cumulative_distance = 0

            if i == len(stroke) - 1:
                adjusted_stroke.append(point_b)

        return adjusted_stroke

    def __str__(self) -> str:
        return (
            f"PointDensityAugmentation(min_length_fraction={self.min_length_fraction}, inplace={self.inplace})"
        )


class GaussianNoise(BaseTransform):

    def __init__(self, window_size: int = 3, p: float = 0.5, inplace: bool = False):
        self.window_size = window_size
        self.inplace = inplace
        self.mean = 0
        self.p = p

    def __call__(self, doodle: DoodleObject) -> DoodleObject:

        if random.random() > self.p:
            return doodle

        if not self.inplace:
            doodle = copy.deepcopy(doodle)

        bbox = get_bbox(doodle)
        box_size = max(bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)
        std_dev = box_size / 100

        doodle.strokes = [
            self._add_noise_to_stroke(stroke, std_dev)
            for stroke in doodle.strokes
        ]
        return doodle

    def _add_noise_to_stroke(self, stroke: List[Tuple[float, float]], std_dev: float) -> List[Tuple[float, float]]:

        if len(stroke) <= 2:
            return stroke

        noisy_middle = [
            (x + random.gauss(self.mean, std_dev), y + random.gauss(self.mean, std_dev))
            for x, y in stroke[1:-1]
        ]

        noisy_stroke = [stroke[0]] + noisy_middle + [stroke[-1]]
        return self._moving_average_smooth(noisy_stroke)

    def _moving_average_smooth(self, stroke: List[Tuple[float, float]]) -> List[Tuple[float, float]]:

        if len(stroke) < self.window_size:
            return stroke

        smoothed = [stroke[0]]

        for i in range(1, len(stroke) - 1):
            window = stroke[
                     max(0, i - self.window_size // 2): min(len(stroke), i + self.window_size // 2 + 1)
                     ]
            x_vals, y_vals = zip(*window)
            smoothed.append((sum(x_vals) / len(x_vals), sum(y_vals) / len(y_vals)))

        smoothed.append(stroke[-1])
        return smoothed

    def __str__(self) -> str:
        return f"GaussianNoise(window_size={self.window_size}, inplace={self.inplace})"


def rescale_sketch(sketch, size, padding) -> List:
    bbox = get_bbox(sketch)
    x_min, y_min, x_max, y_max = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    width = x_max - x_min
    height = y_max - y_min
    effective_n = size - 2 * padding
    scaling_factor = effective_n - 1
    max_dim = max(width, height) + 0.001  ###
    scale = scaling_factor / max_dim

    if width > height:
        vertical_padding = (effective_n - height * scale) / 2
        horizontal_padding = 0
    else:
        horizontal_padding = (effective_n - width * scale) / 2
        vertical_padding = 0

    horizontal_padding += padding
    vertical_padding += padding
    scaled_sketch = [
        [
            (
                int(round((x - x_min) * scale + horizontal_padding)),
                int(round((y - y_min) * scale + vertical_padding))
            )
            for x, y in line
        ]
        for line in sketch
    ]

    return scaled_sketch


def plot_sketch(sketch: List, size: int, padding: int = 1) -> np.array:
    scaled_sketch = rescale_sketch(sketch, size, padding)
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    for line in scaled_sketch:
        draw.line(line, fill=0, width=1)

    return np.array(img)


def plot_strokes(sketch: List, size: int, padding: int = 1) -> List:
    strokes = []
    scaled_sketch = rescale_sketch(sketch, size, padding)
    for line in scaled_sketch:
        img = Image.new("L", (size, size), color=255)
        draw = ImageDraw.Draw(img)
        draw.line(line, fill=0, width=1)
        strokes.append(np.array(img))

    return strokes


def plot_strokes_full(sketch: List, size: int, padding: int = 1) -> List:
    strokes = []
    for stroke in sketch:
        try:
            scaled_stroke = rescale_sketch([stroke], size, padding)[0]
        except Exception as e:
            print(stroke)

        img = Image.new("L", (size, size), color=255)
        draw = ImageDraw.Draw(img)
        draw.line(scaled_stroke, fill=0, width=1)
        strokes.append(np.array(img))

    return strokes


def normalize_strokes(stroke_images: List, size: int, max_stroke_num: int) -> torch.Tensor:
    stroke_images = np.array(stroke_images)
    stroke_images = torch.tensor(stroke_images, dtype=torch.float32)
    stroke_images /= 255
    current_stroke_num = stroke_images.size(0)
    if current_stroke_num < max_stroke_num:
        pad = max_stroke_num - current_stroke_num
        padding_tensor = torch.zeros((pad, size, size), dtype=torch.float32)
        stroke_images = torch.cat([stroke_images, padding_tensor], dim=0)

    return stroke_images


def normalize_sketch(sketch: List) -> torch.Tensor:
    sketch = np.array(sketch)
    sketch = torch.tensor(sketch, dtype=torch.float32)
    sketch /= 255
    return sketch
