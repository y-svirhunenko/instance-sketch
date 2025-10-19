import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from matplotlib import colors as mcolors
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import ConcatDataset, DataLoader
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import opts
from dataset.doodle_transforms import normalize_strokes, plot_strokes
from detector_pipeline.detector_dataset import *
from detector_pipeline.detector_pipeline_utils import *
from models.sketch_transformer import *
from utils.log_utils import *
from utils.utils import *


custom_colors = [
    (220, 80, 80),  # Muted Red
    (80, 120, 220),  # Muted Blue
    (230, 150, 190),  # Soft Pink
    (240, 230, 140),  # Soft Yellow
    (120, 200, 120),  # Soft Green
    (160, 120, 180),  # Muted Purple
    (240, 180, 100)  # Soft Orange
]


def plot_yolo_custom_colors(results):
    """
    Plots YOLO detections with custom bounding box colors.
    Supports multiple bboxes in one image.
    Returns both:
        - original image (no bboxes)
        - annotated image (with bboxes)
    """

    result = results
    orig_img = result.orig_img.copy()
    img_annotated = orig_img.copy()
    annotator = Annotator(img_annotated, line_width=2)

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cls = classes[i]
        conf = confs[i]
        color = custom_colors[cls % len(custom_colors)]
        label = f"{result.names[cls]} {conf:.2f}"
        annotator.box_label((x1, y1, x2, y2), label, color=color)

    # Return original and annotated
    return orig_img, annotator.result()


class DetectorPipelineTester:
    def __init__(self, test_loader, options, device, model, clustering_model=None, experiment_logging=False):

        self.opt = options
        self.device = device
        self.model = model
        self.test_loader = test_loader
        self.doodle_meta = DoodleDatasetDetectorMeta()
        self.clustering_model = clustering_model
        self.segmentation_labels = (
            self.doodle_meta.IND2SEGMENTLABEL_EXTEND
            if clustering_model is not None
            else self.doodle_meta.IND2SEGMENTLABEL
        )
        self.image_visualizer = ImageVisualizerDetector(stroke_labels=self.doodle_meta.IND2SEGMENTLABEL_EXTEND)
        self.visualized_percentage = 0.0
        self.kmeans = KMeans(n_clusters=2, random_state=42)

        self.metric_reporter = MetricsReporter(
            metrics_to_report=["loss", "classification_accuracy", "segmentation_accuracy"],
            f1_scores_to_report=["classification_f1", "segmentation_f1"],
            conf_matrixes_to_report=["classification_matrix", "segmentation_matrix"],
            classification_labels=self.doodle_meta.IND2CLS,
            segmentation_labels=self.segmentation_labels
        )

        self.yolo_model = YOLO('runs/detect/train2/weights/best.pt').to(self.device)
        self.experiment_logging = experiment_logging

    def _map_tensor(self, tensor, mapping):
        out = tensor.clone()
        for i, v in enumerate(out):
            out[i] = mapping.get(v.item(), v.item())
        return out

    def swap_output_labels(self, final_seg_labels):
        for i, label in enumerate(final_seg_labels):
            segm_lab = self.doodle_meta.IND2SEGMENTLABEL[label]
            if segm_lab in self.doodle_meta.IND2SEGMENTLABEL_EXTEND:
                final_seg_labels[i] = self.doodle_meta.SEGMENTLABEL2IND_EXTEND[segm_lab]
        return final_seg_labels

    def find_best_leg_label_flip(self, current_outs, current_labels):
        FRONT_SWITCH = {
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['front-right-foot']:
                self.doodle_meta.SEGMENTLABEL2IND_EXTEND['front-left-foot'],
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['front-left-foot']:
                self.doodle_meta.SEGMENTLABEL2IND_EXTEND['front-right-foot'],
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['front-left-paw']:
                self.doodle_meta.SEGMENTLABEL2IND_EXTEND['front-right-paw'],
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['front-right-paw']:
                self.doodle_meta.SEGMENTLABEL2IND_EXTEND['front-left-paw'],
        }
        BACK_SWITCH = {
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['back-right-foot']:
                self.doodle_meta.SEGMENTLABEL2IND_EXTEND['back-left-foot'],
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['back-left-foot']:
                self.doodle_meta.SEGMENTLABEL2IND_EXTEND['back-right-foot'],
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['back-right-paw']:
                self.doodle_meta.SEGMENTLABEL2IND_EXTEND['back-left-paw'],
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['back-left-paw']:
                self.doodle_meta.SEGMENTLABEL2IND_EXTEND['back-right-paw'],
        }

        def try_flip(outs, mapping):
            flipped = outs.clone()
            for j, v in enumerate(outs):
                if v.item() in mapping:
                    flipped[j] = mapping[v.item()]
            return flipped if (flipped == current_labels).sum() > (outs == current_labels).sum() else outs

        current_outs = try_flip(current_outs, FRONT_SWITCH)
        current_outs = try_flip(current_outs, BACK_SWITCH)
        return current_outs

    def _process_leg_embeddings(self, strokes, lab, cl):

        lab[lab > 3] = 0
        label_map = {
            self.doodle_meta.SEGMENTLABEL2IND['back-foot']: self.doodle_meta.stroke_classes_for_clustering.index(
                'back-foot'),
            self.doodle_meta.SEGMENTLABEL2IND['front-foot']: self.doodle_meta.stroke_classes_for_clustering.index(
                'front-foot'),
            self.doodle_meta.SEGMENTLABEL2IND['back-paw']: self.doodle_meta.stroke_classes_for_clustering.index(
                'back-paw'),
            self.doodle_meta.SEGMENTLABEL2IND['front-paw']: self.doodle_meta.stroke_classes_for_clustering.index(
                'front-paw'),
        }

        plotted = plot_strokes(strokes, self.doodle_meta.CLUSTERING_IMAGE_SIZE)
        normalized = normalize_strokes(plotted, self.doodle_meta.CLUSTERING_IMAGE_SIZE, self.doodle_meta.MAX_STROKE_NUM)
        tensor = normalized.to(self.device).unsqueeze(0)
        lab_padded = self.pad_tensor(lab, target_size=105, dim=0, pad_value=4)

        lab_mapped = self._map_tensor(lab_padded, label_map).unsqueeze(0).to(self.device)
        n = torch.tensor([len(strokes)], device=self.device)

        embeddings = self.clustering_model(tensor, n, lab_mapped)
        return embeddings.squeeze(0)[:len(strokes)].detach().cpu().numpy()

    def change_leg_labels(self, stroke_num, init_strokes, points_num, outs, labels, classes):

        FRONT_TYPES = {self.doodle_meta.SEGMENTLABEL2IND['front-foot'], self.doodle_meta.SEGMENTLABEL2IND['front-paw']}
        BACK_TYPES = {self.doodle_meta.SEGMENTLABEL2IND['back-foot'], self.doodle_meta.SEGMENTLABEL2IND['back-paw']}

        def extract(indices):
            return [init_strokes[i][:points_num[i]].tolist() for i in indices]

        def cluster(legs, lab, cl):
            if not legs:
                return []
            emb = self._process_leg_embeddings(legs, lab, cl)
            return self.kmeans.fit_predict(emb) if len(emb) > 1 else [0] * len(emb)

        front_idx = [i for i, l in enumerate(outs) if l.item() in FRONT_TYPES]
        back_idx = [i for i, l in enumerate(outs) if l.item() in BACK_TYPES]

        front_clusters = cluster(extract(front_idx), outs[front_idx], classes)
        back_clusters = cluster(extract(back_idx), outs[back_idx], classes)

        f_ptr, b_ptr = 0, 0
        for i in range(stroke_num):
            label = outs[i].item()
            if label in FRONT_TYPES:
                orient = 'right' if front_clusters[f_ptr] == 0 else 'left'
                new_label = self.doodle_meta.IND2SEGMENTLABEL[label].replace('-', f'-{orient}-')
                outs[i] = self.doodle_meta.SEGMENTLABEL2IND_EXTEND[new_label]
                f_ptr += 1
            elif label in BACK_TYPES:
                orient = 'right' if back_clusters[b_ptr] == 0 else 'left'
                new_label = self.doodle_meta.IND2SEGMENTLABEL[label].replace('-', f'-{orient}-')
                outs[i] = self.doodle_meta.SEGMENTLABEL2IND_EXTEND[new_label]
                b_ptr += 1

        return self.find_best_leg_label_flip(outs, labels)

    def refine_using_clustering(self, stroke_num, init_strokes, points_num, predicted_labels, true_labels,
                                predicted_classes):
        outs = self.swap_output_labels(predicted_labels)
        return self.change_leg_labels(stroke_num, init_strokes, points_num, outs, true_labels, predicted_classes)

    def pad_tensor(self, tensor: torch.Tensor, target_size: int, dim: int = 0, pad_value: float = 0) -> torch.Tensor:

        size_along_dim = tensor.size(dim)
        if size_along_dim > target_size:
            return tensor.narrow(dim, 0, target_size)

        pad_len = target_size - size_along_dim
        if pad_len == 0:
            return tensor

        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_len
        pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)

        return torch.cat([tensor, pad_tensor], dim=dim)

    def process_image_with_yolo_and_segment(self, instance, model_outputs):

        img = instance['image'].squeeze(0).to(self.device)
        detector_strokes = instance['detector_strokes'].squeeze(0).to(self.device)
        strokes = instance['strokes'].squeeze(0).to(self.device)
        stroke_locations = instance['stroke_locations'].squeeze(0).to(self.device)
        labels = instance['seg_label'].squeeze(0).to(self.device)
        labels_ext = instance['seg_label_extend'].squeeze(0).to(self.device)
        stroke_point_num = instance['stroke_point_number'].squeeze(0).to(self.device)
        stroke_cls = instance['stroke_classes'].squeeze(0).to(self.device)
        init_strokes = instance['initial_strokes'].squeeze(0).to(self.device)
        point_num = instance['stroke_point_number'].squeeze(0).to(self.device)

        image_rgb = np.array(T.ToPILImage()(img).convert("RGB"))
        yolo_results = \
        self.yolo_model.predict(source=image_rgb, save=False, imgsz=512, conf=0.25, iou=0.45, device=self.device)[0]
        boxes, class_ids = yolo_results.boxes.xyxy.cpu().numpy(), yolo_results.boxes.cls.cpu().numpy()
        valid_strokes = [detector_strokes[i][:stroke_point_num[i].item()] for i in range(len(detector_strokes))]

        matched = set()
        pred_labels, true_labels = [], []
        pred_cls, true_cls = [], []

        stroke_to_bbox = {}

        for s_idx, pts in enumerate(valid_strokes):
            max_points_in_bbox = 0
            best_bbox_idx = None
            for b_idx, (x1, y1, x2, y2) in enumerate(boxes):
                points_in_box = ((x1 <= pts[:, 0]) & (pts[:, 0] <= x2) &
                                 (y1 <= pts[:, 1]) & (pts[:, 1] <= y2)).sum()
                if points_in_box > max_points_in_bbox:
                    max_points_in_bbox = points_in_box
                    best_bbox_idx = b_idx
            if best_bbox_idx is not None:
                stroke_to_bbox[s_idx] = best_bbox_idx

        bbox_to_strokes = {}
        for s_idx, b_idx in stroke_to_bbox.items():
            bbox_to_strokes.setdefault(b_idx, []).append(s_idx)

        for b_idx, stroke_idx in bbox_to_strokes.items():
            x1, y1, x2, y2 = boxes[b_idx]
            cls_id = class_ids[b_idx]
            matched.update(stroke_idx)

            st = torch.stack([strokes[i] for i in stroke_idx]).unsqueeze(0)
            sl = torch.stack([stroke_locations[i] for i in stroke_idx]).unsqueeze(0)
            sel_strokes = self.pad_tensor(st, target_size=105, dim=1, pad_value=0)
            sel_locs = self.pad_tensor(sl, target_size=105, dim=1, pad_value=0)

            stroke_num = torch.tensor([len(stroke_idx)], device=self.device)
            category = torch.tensor([int(cls_id)], device=self.device)
            true_labels_e = labels_ext[stroke_idx].to(self.device)

            true_cls.extend(stroke_cls[stroke_idx].tolist())
            pred_cls.extend([int(cls_id)] * len(stroke_idx))

            seg_logits = self.model(strokes=sel_strokes, stroke_locations=sel_locs, sketch_stroke_num=stroke_num,
                                    sketch_category=category).squeeze(0)
            predicted = torch.argmax(seg_logits, dim=1)[:stroke_num]

            init_sel = torch.stack([init_strokes[i] for i in stroke_idx]).to(self.device)
            points_sel = point_num[stroke_idx].to(self.device)
            predicted = self.refine_using_clustering(stroke_num, init_sel, points_sel, predicted, true_labels_e,
                                                     category)

            pred_labels.extend(predicted.cpu().tolist())
            true_labels.extend(true_labels_e.cpu().tolist())

        all_idx = set(range(len(labels)))
        for idx in all_idx - matched:
            model_outputs.normalized_segmentation_true.append(labels[idx].item())
            model_outputs.normalized_segmentation_predicted.append(-1)
            model_outputs.normalized_logits_true.append(stroke_cls[idx].item())
            model_outputs.normalized_logits_predicted.append(-1)

        model_outputs.normalized_segmentation_predicted.extend(pred_labels)
        model_outputs.normalized_segmentation_true.extend(true_labels)
        model_outputs.normalized_logits_true.extend(true_cls)
        model_outputs.normalized_logits_predicted.extend(pred_cls)

        choose = random.choices([True, False], [self.visualized_percentage, 1 - self.visualized_percentage])[0]
        if choose and self.experiment_logging:
            orig_img, image_with_boxes = plot_yolo_custom_colors(yolo_results)
            strokes = [detector_strokes[i][:stroke_point_num[i].item()].cpu().tolist() for i in matched]
            point_nums = [stroke_point_num[i].item() for i in matched]
            batch_plot_images = self.image_visualizer.plot_result(strokes, point_nums, pred_labels, true_labels)  #####
            model_outputs.all_plot_images.extend(batch_plot_images)
            model_outputs.all_plot_images.append(image_with_boxes)
            model_outputs.all_plot_images.append(orig_img)

        return model_outputs

    def get_model_outputs_class(self):
        num_labels = (
            self.doodle_meta.NUM_STROKE_LABELS_EXTEND
            if self.clustering_model is not None
            else self.doodle_meta.NUM_STROKE_LABELS
        )
        return ModelOutputs(self.doodle_meta.NUM_SKETCH_CLASSES, num_labels)

    def test(self, max_batches=2000):
        self.model.eval()
        outputs = self.get_model_outputs_class()
        count = 0
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if max_batches and i >= max_batches:
                    break
                outputs = self.process_image_with_yolo_and_segment(batch, outputs)
                count = i
            print(f"Instances processed: {count + 1}")
        return outputs

    def calculate_f1_score(self, true, predicted):

        if not true or not predicted or len(true) != len(predicted):
            return 0.0

        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        classes = set(true)

        for t, p in zip(true, predicted):
            if p == t:
                tp[t] += 1
            else:
                fn[t] += 1
                fp[p] += 1

        f1_scores = []
        for c in classes:
            precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0
            recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0
            if precision + recall == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
        return sum(f1_scores) / len(f1_scores)

    def calculate_accuracy(self, true, predicted):

        correct = 0
        total = 0
        for t, p in zip(true, predicted):
            total += 1
            if p == -1:
                continue
            if t == p:
                correct += 1

        return correct / total if total > 0 else 0

    def validate(self):

        outputs = self.test()
        seg_acc = self.calculate_accuracy(outputs.normalized_segmentation_true,
                                          outputs.normalized_segmentation_predicted)
        cls_acc = self.calculate_accuracy(outputs.normalized_logits_true, outputs.normalized_logits_predicted)
        seg_f1 = self.calculate_f1_score(outputs.normalized_segmentation_true,
                                         outputs.normalized_segmentation_predicted)
        cls_f1 = self.calculate_f1_score(outputs.normalized_logits_true, outputs.normalized_logits_predicted)

        print(f'Classification f1 score: {cls_f1}')
        print(f'Segmentation f1 score: {seg_f1}')
        print(f"Classification Accuracy: {cls_acc}")
        print(f"Segmentation Accuracy: {seg_acc}")

        if self.experiment_logging:
            self.metric_reporter.report_plots(1, 'Validation', outputs)


def get_detector_dataset(dataset_types: List, opt: Dict[str, Any] = None, transforms: TransformsCompose = None,
                         mode: str = None, multiplication_factor: int = None) -> Any:
    datasets = []
    for d_type in dataset_types:
        if d_type == 'xml':
            dataset_xml = DoodleDatasetXMLDetector(data_dir=opt['dataset_path_xml'],
                                                   split_file_name=opt['data_split_xml'],
                                                   mode=mode, transform=transforms)
            datasets.append(dataset_xml)
        elif d_type == 'json':
            dataset_json = DoodleDatasetJSONDetector(data_dir=opt['dataset_path'],
                                                     split_file_name=opt['data_split_json'],
                                                     mode=mode, transform=transforms)
            datasets.append(dataset_json)

    if not datasets:
        return None

    if multiplication_factor is not None:
        datasets *= multiplication_factor
        return ConcatDataset(datasets)

    if len(dataset_types) >= 2:
        return ConcatDataset(datasets)
    else:
        return datasets[0]


def create_dataset(dataloader, out_dir="yolo_dataset", instances=None, split="train"):
    out_dir = Path(out_dir)
    img_dir = out_dir / "images" / split
    label_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    dataset_len = len(dataloader)

    to_pil = T.ToPILImage()
    if not instances:
        instances = 100

    for j in range(instances):

        i = j % dataset_len
        sample = dataloader[i]
        img_tensor = sample["image"]  # (3, H, W), torch.float32
        labels = sample["labels"]  # (N, 5) where 5 = [cls, x, y, w, h]

        # Save image
        img = to_pil(img_tensor)
        img_name = f"{split}_{i:05d}.jpg"
        img_path = img_dir / img_name
        img.save(img_path)

        # Save labels
        label_lines = []
        for label in labels:
            cls, x, y, w, h = label.tolist()
            if sum([x, y, w, h]) == 0:
                continue
            label_lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        label_path = label_dir / f"{img_name.replace('.jpg', '.txt')}"
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))


def create_data_yaml(dataset_root="yolo_dataset", num_classes=7, class_names=None):
    dataset_root = Path(dataset_root)
    yaml_path = dataset_root / "data.yaml"

    class_names = class_names or [f"class_{i}" for i in range(num_classes)]

    content = f"""\
    train: images/train
    val: images/val

    nc: {num_classes}
    names: {class_names}
    """

    with open(yaml_path, "w") as f:
        f.write(content)


class ImageVisualizerDetector:

    def __init__(self, stroke_labels: List):
        self.labels = stroke_labels

    def plot_result(self, strokes: List[List[Tuple[float, float]]], stroke_point_number: List[int],
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

            legend_handle = plt.Line2D([], [], color=color, linewidth=line_width, linestyle='-')
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
        result = img[:, :, :3] if img.shape[2] == 4 else img

        return [result]
