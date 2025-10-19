import random
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors, patheffects as path_effects
import numpy as np
import torch
from clearml import Logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


label_color_map = {
    'wheel': 'gray',
    'frame': 'red',
    'center': cm.get_cmap('Set2')(5),
    'ray': 'orange',
    'leaf': mcolors.to_rgb('#006400'),
    'stem': mcolors.to_rgb('#9ACD32'),
    'steam': 'brown',
    'leaves': 'green',
    'body': cm.get_cmap('tab10')(0),
    'petal': cm.get_cmap('tab10')(6),
    'neck': cm.get_cmap('tab10')(4),
    'head': cm.get_cmap('tab10')(6),
    'tail': 'green',

    'back-right-foot': cm.get_cmap('tab20b')(1),
    'back-right-leg': cm.get_cmap('tab20b')(3),
    'back-left-foot': cm.get_cmap('tab20b')(17),
    'back-left-leg': cm.get_cmap('tab20b')(19),
    'front-right-foot': cm.get_cmap('tab20b')(13),
    'front-right-leg': cm.get_cmap('tab20b')(15),
    'front-left-foot': cm.get_cmap('tab20b')(5),
    'front-left-leg': cm.get_cmap('tab20b')(7),
}


class ImageVisualizer:

    def __init__(self, percentage: float, stroke_labels: List):
        self.percentage = percentage
        self.labels = stroke_labels

    def plot_sketch(self, strokes: List[List[Tuple[float, float]]], stroke_point_number: List[int],
                    recognised_label: List[int], correct_label: List[int]) -> np.ndarray:

        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.axis('off')
        num_strokes = len(correct_label)
        handles = []
        labels = []

        def label_name(idx):
            if idx == -1:
                return "undetected"
            return self.labels[idx].replace("paw", "leg")

        unique_labels = list(
            set(label_name(label) for label in correct_label + [l for l in recognised_label if l != -1]))
        tab10_cmap = cm.get_cmap('tab10')

        label_to_color = {}
        tab10_index = 0
        for label in sorted(unique_labels):
            if label in label_color_map:
                label_to_color[label] = label_color_map[label]
            else:
                label_to_color[label] = tab10_cmap(tab10_index % tab10_cmap.N)
                tab10_index += 1

        for i in range(num_strokes):
            stroke_points = strokes[i][:stroke_point_number[i]]
            x_points, y_points = zip(*stroke_points)

            rec_label = label_name(recognised_label[i])
            cor_label = label_name(correct_label[i])
            is_misclassified = (rec_label != cor_label) or (recognised_label[i] == -1)

            label = (
                f"{i + 1}: Recognized: {rec_label} | Correct: {cor_label}"
                if is_misclassified else
                f"{i + 1}: {cor_label}"
            )

            color = 'black' if is_misclassified else label_to_color[cor_label]
            line_style = '--' if is_misclassified else '-'
            line_width = 2

            ax.plot(x_points, y_points, color=color, linewidth=line_width, linestyle=line_style)

            mid_idx = len(x_points) // 2
            text = ax.text(
                x_points[mid_idx], y_points[mid_idx], str(i + 1),
                fontsize=12, fontweight='bold', color='black',
                ha='center', va='center'
            )
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='white'),
                path_effects.Normal()
            ])

            legend_handle = plt.Line2D([], [], color=color, linewidth=line_width, linestyle=line_style)
            handles.append(legend_handle)
            labels.append(label)

        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small',
                  title="Recognized strokes", borderaxespad=0.)
        plt.tight_layout()
        img_buffer = BytesIO()
        plt.savefig(img_buffer, bbox_inches='tight', pad_inches=0.1, dpi=100, format='png')
        plt.close()
        img_buffer.seek(0)
        img = plt.imread(img_buffer)

        return img[:, :, :3] if img.shape[2] == 4 else img

    def plot_correct_and_wrong_strokes(self, data_batch: Dict[str, Any], recognised_labels: List[int],
                                       correct_labels: List[int]) -> List[np.ndarray]:

        input_strokes = data_batch['initial_strokes']
        stroke_point_num = data_batch['stroke_point_number']
        sketch_stroke_num = data_batch['stroke_number']

        pos = 0
        plot_images = []
        for i in range(len(sketch_stroke_num)):
            number = int(sketch_stroke_num[i])
            choose = random.choices([True, False], [self.percentage, 1 - self.percentage])[0]
            if choose:
                strokes = input_strokes[i][:number].tolist()
                stroke_point_number = stroke_point_num[i][:number].tolist()
                recognised = recognised_labels[pos:pos + number]
                correct = correct_labels[pos:pos + number]
                img_array = self.plot_sketch(strokes, stroke_point_number, recognised, correct)
                plot_images.append(img_array)

            pos += number
        return plot_images


class ModelOutputs:
    def __init__(self, classification_labels_len: int = 0, stroke_segmentation_labels_len: int = 0):
        self.classification_labels = list(range(classification_labels_len))
        self.stroke_classification_labels = list(range(stroke_segmentation_labels_len))

        self.loss = 0
        self.correct_logits = 0
        self.correct_segmentation = 0
        self.correct_segmentation = 0
        self.total_number_of_sketches = 0
        self.total_number_of_strokes = 0
        self.raw_logits = []
        self.raw_segmentation = []
        self.raw_segmentation = []
        self.normalized_logits_predicted = []
        self.normalized_logits_true = []
        self.normalized_segmentation_predicted = []
        self.normalized_segmentation_true = []
        self.all_plot_images = []

    def raw_outputs_to_str(self) -> str:
        return (f"Logits:\n{self.raw_logits}\n"
                f"Segmentation:\n{self.raw_segmentation}")

    def raw_outputs_to_dict(self, digits: int = None) -> dict:
        output_data = {
            "Logits": self.raw_logits,
            "Segmentation": self.raw_segmentation
        }
        if digits is not None:
            output_data["Logits"] = [[round(float(logit), digits) for logit in logits_list] for logits_list in
                                     self.raw_logits]
            output_data["Segmentation"] = [
                [[round(float(enc), digits) for enc in enc_list] for enc_list in enc_structure] for enc_structure in
                self.raw_segmentation]

        return output_data

    def get_classification_accuracy(self) -> float:
        return self.correct_logits / self.total_number_of_sketches

    def get_segmentation_accuracy(self) -> float:
        return self.correct_segmentation / self.total_number_of_strokes


class MetricsReporter:

    def __init__(self, metrics_to_report: list[str] = None, f1_scores_to_report: list[str] = None,
                 conf_matrixes_to_report: list[str] = None, average: str = 'macro',
                 classification_labels: list[str] = None, segmentation_labels: list[str] = None):

        self.metrics_to_report = metrics_to_report
        self.f1_scores_to_report = f1_scores_to_report
        self.conf_matrixes_to_report = conf_matrixes_to_report
        self.average = average
        self.classification_labels = classification_labels
        self.segmentation_labels = segmentation_labels
        self.logger = Logger.current_logger()

    def calculate_f1_scores(self, model_outputs: ModelOutputs) -> dict[str, float | None]:

        scores = {}
        if self.f1_scores_to_report is None:
            return scores

        if "classification_f1" in self.f1_scores_to_report:
            scores["classification_f1"] = self._calculate_f1(
                model_outputs.normalized_logits_true, model_outputs.normalized_logits_predicted)
        if "segmentation_f1" in self.f1_scores_to_report:
            scores["segmentation_f1"] = self._calculate_f1(
                model_outputs.normalized_segmentation_true,
                model_outputs.normalized_segmentation_predicted)

        return scores

    def calculate_confusion_matrices(self, model_outputs: ModelOutputs) -> dict[str, np.ndarray | None]:

        matrices = {}
        if self.conf_matrixes_to_report is None:
            return matrices

        if "classification_matrix" in self.conf_matrixes_to_report:
            matrices["classification"] = self._calculate_confusion_matrix(
                model_outputs.normalized_logits_true,
                model_outputs.normalized_logits_predicted,
                model_outputs.classification_labels)
        if "segmentation_matrix" in self.conf_matrixes_to_report:
            matrices["segmentation"] = self._calculate_confusion_matrix(
                model_outputs.normalized_segmentation_true,
                model_outputs.normalized_segmentation_predicted,
                model_outputs.stroke_classification_labels)

        return matrices

    def report_graphs(self, epoch: int, series: str, model_outputs: ModelOutputs, trial_number: int = None):

        if self.metrics_to_report is not None:
            metric_titles = {
                "loss": "Loss",
                "classification_accuracy": "Classification Accuracy",
                "segmentation_accuracy": "Segmentation Accuracy"
            }

            if trial_number is not None:
                for key in metric_titles:
                    metric_titles[key] = f'Trial {trial_number}: {metric_titles[key]}'

            if "loss" in self.metrics_to_report and model_outputs.loss is not None:
                self.logger.report_scalar(title=metric_titles["loss"], series=series, iteration=epoch,
                                          value=model_outputs.loss)

            if "classification_accuracy" in self.metrics_to_report:
                acc = model_outputs.get_classification_accuracy()
                self.logger.report_scalar(title=metric_titles["classification_accuracy"], series=series,
                                          iteration=epoch, value=acc)

            if "segmentation_accuracy" in self.metrics_to_report:
                acc = model_outputs.get_segmentation_accuracy()
                self.logger.report_scalar(title=metric_titles["segmentation_accuracy"], series=series,
                                          iteration=epoch, value=acc)

    def report_f1_scores(self, epoch: int, series: str, model_outputs: ModelOutputs, trial_number: int = None):

        if self.f1_scores_to_report is not None:
            f1_dict = self.calculate_f1_scores(model_outputs)
            f1_titles = {
                "classification_f1": "F1 Score (Classification)",
                "segmentation_f1": "F1 Score (Segmentation)"
            }

            if trial_number is not None:
                for key in f1_titles:
                    f1_titles[key] = f'Trial {trial_number}: {f1_titles[key]}'

            for key in self.f1_scores_to_report:
                if key in f1_dict and f1_dict[key] is not None:
                    self.logger.report_scalar(title=f1_titles[key], series=series, iteration=epoch, value=f1_dict[key])

    def report_plots(self, epoch: int, series: str, model_outputs: ModelOutputs, trial_number: int = None):

        cm_dict = self.calculate_confusion_matrices(model_outputs)
        metric_names = {
            "classification": "Confusion Matrix (Classification)",
            "segmentation": "Confusion Matrix (Segmentation)"
        }

        if trial_number is not None:
            for key in metric_names:
                metric_names[key] = f'Trial {trial_number}: {metric_names[key]}'

        for key, matrix in cm_dict.items():
            if matrix is None:
                continue
            xlabels = ylabels = self.classification_labels if key == "classification" else self.segmentation_labels

            self.logger.report_confusion_matrix(
                title=metric_names[key],
                iteration=epoch,
                matrix=matrix,
                series=series,
                xlabels=xlabels,
                ylabels=ylabels
            )

        for i, image in enumerate(model_outputs.all_plot_images):
            self.logger.report_image(title=series, series='Image ' + str(i), iteration=epoch, image=image)

    def _calculate_f1(self, true: list, predicted: list) -> float:
        if self.average not in ['macro', 'micro', 'weighted']:
            print('Invalid F1 Score! Choosing macro')
            self.average = 'macro'
        return f1_score(true, predicted, average=self.average)

    def _calculate_confusion_matrix(self, true: list, predicted: list, labels: list) -> np.ndarray:
        return confusion_matrix(true, predicted, labels=labels)

    def calculate_class_accuracy(self, valid_model_outputs: ModelOutputs = None) -> None:

        true_labels = valid_model_outputs.normalized_logits_true
        predicted_labels = valid_model_outputs.normalized_logits_predicted
        labels = valid_model_outputs.classification_labels

        report = classification_report(
            true_labels, predicted_labels, labels=labels, output_dict=True, zero_division=0
        )

        class_metrics = [
            (
                self.classification_labels[label],
                report[str(label)]['recall'],
                report[str(label)]['f1-score']
            )
            for label in labels if str(label) in report
        ]
        class_metrics.sort(key=lambda x: x[1], reverse=True)

        print("Class Accuracy and F1 Scores")
        print(f"{'Class Name':<20} {'Recall':<10} {'F1 Score'}")
        print("-" * 45)

        for class_name, recall, f1 in class_metrics:
            print(f"{class_name:<20} {recall:.4f}     {f1:.4f}")

    def calculate_segmentation_accuracy(self, valid_model_outputs: ModelOutputs = None) -> None:

        true_labels = valid_model_outputs.normalized_segmentation_true
        predicted_labels = valid_model_outputs.normalized_segmentation_predicted
        labels = valid_model_outputs.stroke_classification_labels
        report = classification_report(true_labels, predicted_labels, labels=labels, output_dict=True, zero_division=0)

        segmentation_metrics = [
            (
                self.segmentation_labels[label],
                report[str(label)]['recall'],
                report[str(label)]['f1-score']
            )
            for label in labels if str(label) in report
        ]
        segmentation_metrics.sort(key=lambda x: x[1], reverse=True)

        print("Segmentation Accuracy and F1 Scores")
        print(f"{'Segment Name':<20} {'Recall':<10} {'F1 Score'}")
        print("-" * 45)

        for segment_name, recall, f1 in segmentation_metrics:
            print(f"{segment_name:<20} {recall:.4f}     {f1:.4f}")


def compute_dataset_statistics(test_loader, doodle_meta, multiplier=None):
    class_counter = defaultdict(int)
    label_counter = defaultdict(int)

    print('Dataset oversampling factors:')
    if multiplier is None:
        print("None (no oversampling applied)")
    else:
        print(multiplier)
    print()

    for data_batch in test_loader:
        class_labels = data_batch['category']
        for i in range(len(class_labels)):
            class_counter[class_labels[i].item()] += 1

        if 'stroke_number' in data_batch and 'seg_label' in data_batch:
            sketch_stroke_num = data_batch['stroke_number']
            labels = data_batch['seg_label']
            for i, num_strokes in enumerate(sketch_stroke_num):
                current_labels = labels[i][:num_strokes]
                for label in current_labels:
                    label_counter[label.item()] += 1

    print('Class statistics after oversampling:')
    for i in range(len(doodle_meta.IND2CLS)):
        print(f'{doodle_meta.IND2CLS[i]}: {class_counter[i]}')
    print()

    if label_counter:
        print('Label statistics after oversampling:')
        for i in range(len(doodle_meta.IND2SEGMENTLABEL)):
            print(f'{doodle_meta.IND2SEGMENTLABEL[i]}: {label_counter[i]}')
        print()

    readable_class_counter = {
        doodle_meta.IND2CLS[i]: class_counter[i]
        for i in range(len(doodle_meta.IND2CLS))
    }

    readable_label_counter = {
        doodle_meta.IND2SEGMENTLABEL[i]: label_counter[i]
        for i in range(len(doodle_meta.IND2SEGMENTLABEL))
    }

    class_counts = [class_counter[i] for i in range(len(doodle_meta.IND2CLS))]
    label_counts = [label_counter[i] for i in range(len(doodle_meta.IND2SEGMENTLABEL))]

    return readable_class_counter, readable_label_counter, class_counts, label_counts


def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
