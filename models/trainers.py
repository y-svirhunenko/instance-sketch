import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from tqdm import tqdm
from sklearn.cluster import KMeans

from dataset.doodle_dataset import DoodleDatasetMeta
from dataset.doodle_transforms import normalize_strokes, plot_strokes
from models.losses import *
from models.sketch_transformer import *
from utils.log_utils import *
from utils.utils import *



class SegmentatorTrainer:

    def __init__(self, train_loader, test_loader, model, options, segmentator_opt, clustering_model=None):

        self.opt = options
        self.device = torch.device(f"cuda:{options['gpu']}" if options['device'] == "gpu" else "cpu")
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optim = torch.optim.Adam(model.parameters(), segmentator_opt['learning_rate'],
                                      weight_decay=segmentator_opt['weight_decay'])
        self.scheduler = CosineAnnealingWarmRestarts(self.optim, T_0=100)
        self.doodle_meta = DoodleDatasetMeta()
        self.criterions = self.get_criterions()
        self.max_epoch = self.opt['max_epochs']
        self.best_f1 = 0
        self.iter_test = 3
        self.save_checkpoint = True  # False

        self.kmeans = KMeans(n_clusters=2, random_state=42)
        self.clustering_model = clustering_model
        self.segmentation_labels = self.doodle_meta.IND2SEGMENTLABEL_EXTEND if self.clustering_model is not None else self.doodle_meta.IND2SEGMENTLABEL
        self.image_visualizer = None  # ImageVisualizer(percentage=0.1, stroke_labels=self.segmentation_labels)

        self.metric_reporter = MetricsReporter(
            metrics_to_report=["loss", "segmentation_accuracy"],
            f1_scores_to_report=["segmentation_f1"],
            conf_matrixes_to_report=["segmentation_matrix"],
            segmentation_labels=self.segmentation_labels)

        if self.save_checkpoint:
            self.file_paths = FilePaths(self.opt)
            self.file_paths.check_all()

    def get_criterions(self) -> Dict[str, Any]:

        _, _, _, stats_labels = compute_dataset_statistics(self.train_loader, self.doodle_meta,
                                                           self.doodle_meta.class_multipliers)
        label_weights = get_class_weights(stats_labels).to(self.device)

        print('Cross-entropy loss is used')
        classification_loss = nn.CrossEntropyLoss().to(self.device)
        segmentation_loss = nn.CrossEntropyLoss(weight=label_weights).to(self.device)

        criterions = {
            'classification_criterion': classification_loss,
            'segmentation_criterion': segmentation_loss
        }
        return criterions

    def calculate_loss(self, data_batch: Dict[str, Any], seg_outs: torch.Tensor = None) -> torch.Tensor:

        sketch_stroke_num = data_batch['stroke_number'].to(self.device)
        seg_labels = data_batch['seg_label'].to(self.device)
        seg_labels = seg_labels.squeeze(-1)
        batch_size, max_strokes = sketch_stroke_num.size(0), seg_outs.size(1)
        mask = torch.arange(max_strokes, device=self.device).expand(batch_size,
                                                                    max_strokes) < sketch_stroke_num.unsqueeze(1)
        seg_outs = seg_outs[mask]
        seg_labels = seg_labels[mask]

        loss = self.criterions['segmentation_criterion'](seg_outs, seg_labels)
        return loss

    def get_seg_outs_and_labels(self, seg_outs: torch.Tensor, seg_label: torch.Tensor,
                                sketch_stroke_num: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, max_strokes = sketch_stroke_num.size(0), seg_outs.size(1)
        mask = torch.arange(max_strokes).expand(batch_size, max_strokes) < sketch_stroke_num.unsqueeze(1)
        _, top_labels = torch.topk(seg_outs, k=1, dim=-1)
        top_labels = top_labels.squeeze(-1)
        masked_top_labels = top_labels[mask]
        masked_seg_labels = seg_label[mask]

        return masked_top_labels, masked_seg_labels

    def swap_output_labels(self, final_seg_labels: torch.Tensor) -> torch.Tensor:

        for i, label in enumerate(final_seg_labels):
            segm_lab = self.doodle_meta.IND2SEGMENTLABEL[label]
            if segm_lab in self.doodle_meta.IND2SEGMENTLABEL and segm_lab in self.doodle_meta.IND2SEGMENTLABEL_EXTEND:
                label = self.doodle_meta.IND2SEGMENTLABEL[label]
                label = self.doodle_meta.SEGMENTLABEL2IND_EXTEND[label]
                final_seg_labels[i] = label

        return final_seg_labels

    def find_best_leg_label_flip(self, current_outs, current_labels):

        FRONT_SWITCH = {
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['front-right-foot']: self.doodle_meta.SEGMENTLABEL2IND_EXTEND[
                'front-left-foot'],
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['front-left-foot']: self.doodle_meta.SEGMENTLABEL2IND_EXTEND[
                'front-right-foot'],
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['front-left-paw']: self.doodle_meta.SEGMENTLABEL2IND_EXTEND[
                'front-right-paw'],
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['front-right-paw']: self.doodle_meta.SEGMENTLABEL2IND_EXTEND[
                'front-left-paw'],
        }

        BACK_SWITCH = {
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['back-right-foot']: self.doodle_meta.SEGMENTLABEL2IND_EXTEND[
                'back-left-foot'],
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['back-left-foot']: self.doodle_meta.SEGMENTLABEL2IND_EXTEND[
                'back-right-foot'],
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['back-right-paw']: self.doodle_meta.SEGMENTLABEL2IND_EXTEND[
                'back-left-paw'],
            self.doodle_meta.SEGMENTLABEL2IND_EXTEND['back-left-paw']: self.doodle_meta.SEGMENTLABEL2IND_EXTEND[
                'back-right-paw'],
        }

        flipped_outs = current_outs.clone()
        for j in range(len(current_outs)):
            if current_outs[j].item() in FRONT_SWITCH:
                flipped_outs[j] = FRONT_SWITCH[current_outs[j].item()]

        if (current_outs == current_labels).sum() < (flipped_outs == current_labels).sum():
            current_outs = flipped_outs

        flipped_outs = current_outs.clone()
        for j in range(len(current_outs)):
            if current_outs[j].item() in BACK_SWITCH:
                flipped_outs[j] = BACK_SWITCH[current_outs[j].item()]

        if (current_outs == current_labels).sum() < (flipped_outs == current_labels).sum():
            current_outs = flipped_outs
        return current_outs

    def change_leg_labels(self, data_batch: Dict[str, Any], seg_outs: torch.Tensor,
                          final_seg_labels: torch.Tensor) -> torch.Tensor:

        FRONT_TYPES = {
            self.doodle_meta.SEGMENTLABEL2IND['front-foot'],
            self.doodle_meta.SEGMENTLABEL2IND['front-paw']
        }
        BACK_TYPES = {
            self.doodle_meta.SEGMENTLABEL2IND['back-foot'],
            self.doodle_meta.SEGMENTLABEL2IND['back-paw']
        }

        index = 0
        for i, num_strokes in enumerate(data_batch['stroke_number']):

            current_slice = slice(index, index + num_strokes)
            current_outs = seg_outs[current_slice]
            init_strokes = data_batch['initial_strokes'][i]
            point_nums = data_batch['stroke_point_number'][i]

            front_indices = [j for j, x in enumerate(current_outs) if x.item() in FRONT_TYPES]
            front_legs = []
            front_labels = []

            back_indices = [j for j, x in enumerate(current_outs) if x.item() in BACK_TYPES]
            back_legs = []
            back_labels = []

            for j in front_indices:
                stroke = init_strokes[j][:point_nums[j]].tolist()
                front_legs.append(stroke)
                label = self.doodle_meta.stroke_classes_for_clustering.index(
                    self.doodle_meta.IND2SEGMENTLABEL[current_outs[j]])
                front_labels.append(label)

            if front_legs:
                front_emb = self._process_leg_embeddings(front_legs, front_labels)  ###
                front_clusters = self.kmeans.fit_predict(front_emb) if len(front_emb) > 1 else [0] * len(front_emb)

            else:
                front_clusters = []

            for j in back_indices:
                stroke = init_strokes[j][:point_nums[j]].tolist()
                back_legs.append(stroke)
                label = self.doodle_meta.stroke_classes_for_clustering.index(
                    self.doodle_meta.IND2SEGMENTLABEL[current_outs[j]])
                back_labels.append(label)

            if back_legs:
                back_emb = self._process_leg_embeddings(back_legs, back_labels)  ###
                back_clusters = self.kmeans.fit_predict(back_emb) if len(back_emb) > 1 else [0] * len(back_emb)

            else:
                back_clusters = []

            front_ptr = back_ptr = 0
            for j in range(num_strokes):
                label = current_outs[j].item()

                if label in FRONT_TYPES:
                    orientation = 'right' if front_clusters[front_ptr] == 0 else 'left'
                    base_label = self.doodle_meta.IND2SEGMENTLABEL[label]
                    new_label = f"{base_label.replace('-', f'-{orientation}-')}"
                    current_outs[j] = self.doodle_meta.SEGMENTLABEL2IND_EXTEND[new_label]
                    front_ptr += 1

                elif label in BACK_TYPES:
                    orientation = 'right' if back_clusters[back_ptr] == 0 else 'left'
                    base_label = self.doodle_meta.IND2SEGMENTLABEL[label]
                    new_label = f"{base_label.replace('-', f'-{orientation}-')}"
                    current_outs[j] = self.doodle_meta.SEGMENTLABEL2IND_EXTEND[new_label]
                    back_ptr += 1

            current_labels = final_seg_labels[current_slice]
            current_outs = self.find_best_leg_label_flip(current_outs, current_labels)
            seg_outs[index:index + num_strokes] = current_outs
            index += num_strokes

        return seg_outs

    def _process_leg_embeddings(self, strokes, stroke_labels):

        plotted = plot_strokes(strokes, self.doodle_meta.CLUSTERING_IMAGE_SIZE)
        normalized = normalize_strokes(plotted, self.doodle_meta.CLUSTERING_IMAGE_SIZE, self.doodle_meta.MAX_STROKE_NUM)
        tensor = normalized.to(self.device).unsqueeze(0)
        stroke_num = torch.tensor(len(strokes)).to(self.device).unsqueeze(0)
        stroke_labels = torch.tensor(stroke_labels).to(self.device)
        stroke_labels = F.pad(stroke_labels, (0, self.doodle_meta.MAX_STROKE_NUM - stroke_labels.size(0)), value=4)
        stroke_labels = stroke_labels.unsqueeze(0)

        embeddings = self.clustering_model(tensor, stroke_num, stroke_class=stroke_labels).to(self.device)
        result = embeddings.squeeze(0)[:len(strokes)].detach().cpu().numpy()
        return result

    def get_outputs(self, data_batch: Dict[str, Any], seg_outs: torch.Tensor,
                    model_outputs: ModelOutputs, validation: bool = False) -> Tuple[ModelOutputs, List, List]:

        seg_label = data_batch['seg_label_extend' if self.clustering_model is not None else 'seg_label'].to(self.device)
        sketch_stroke_num = data_batch['stroke_number']
        final_seg_outs, final_seg_label = self.get_seg_outs_and_labels(seg_outs, seg_label, sketch_stroke_num)

        if self.clustering_model is not None:
            final_seg_outs = self.swap_output_labels(final_seg_outs)
            final_seg_outs = self.change_leg_labels(data_batch, final_seg_outs, final_seg_label)

        model_outputs.correct_segmentation += (final_seg_outs == final_seg_label).long().sum().item()
        model_outputs.total_number_of_strokes += len(final_seg_label)
        batch_size = seg_label.size(0)
        model_outputs.total_number_of_sketches += batch_size

        if validation:
            model_outputs.normalized_segmentation_predicted.extend(final_seg_outs.tolist())
            model_outputs.normalized_segmentation_true.extend(final_seg_label.tolist())

        return model_outputs, final_seg_outs.tolist(), final_seg_label.tolist()

    def get_model_outputs_class(self) -> ModelOutputs:
        if self.clustering_model is not None:
            model_outputs = ModelOutputs(self.doodle_meta.NUM_SKETCH_CLASSES, self.doodle_meta.NUM_STROKE_LABELS_EXTEND)
        else:
            model_outputs = ModelOutputs(self.doodle_meta.NUM_SKETCH_CLASSES, self.doodle_meta.NUM_STROKE_LABELS)
        return model_outputs

    def test(self) -> ModelOutputs:

        with torch.no_grad():
            self.model.eval()
            model_outputs = self.get_model_outputs_class()

            while True:
                try:
                    for data_batch in self.test_loader:

                        self.model.eval()
                        strokes = data_batch['strokes'].to(self.device)
                        stroke_locations = data_batch['stroke_locations'].to(self.device)
                        sketch_stroke_num = data_batch['stroke_number'].to(self.device)
                        predicted_classes = data_batch['category'].to(self.device)

                        seg_outs = self.model(strokes=strokes,
                                              stroke_locations=stroke_locations,
                                              sketch_stroke_num=sketch_stroke_num,
                                              sketch_category=predicted_classes)

                        loss = self.calculate_loss(data_batch, seg_outs)
                        model_outputs.loss += loss.item()

                        model_outputs, final_seg_outs, final_seg_label = self.get_outputs(
                            data_batch, seg_outs, model_outputs, validation=True)

                        if self.image_visualizer is not None:
                            batch_plot_images = self.image_visualizer.plot_correct_and_wrong_strokes(
                                data_batch, final_seg_outs, final_seg_label
                            )
                            model_outputs.all_plot_images.extend(batch_plot_images)

                    model_outputs.loss /= model_outputs.total_number_of_sketches
                    break

                except RuntimeError as e:
                    if "DataLoader timed out" in str(e):
                        print("DataLoader timeout occurred. Retrying...")
                    else:
                        raise

        return model_outputs

    def train(self) -> None:

        for epoch_id in range(self.max_epoch):
            while True:
                try:
                    print(f"Epoch: {epoch_id + 1}")
                    model_outputs = ModelOutputs()

                    for data_batch in tqdm(self.train_loader):
                        self.model.train()
                        strokes = data_batch['strokes'].to(self.device)
                        stroke_locations = data_batch['stroke_locations'].to(self.device)
                        sketch_stroke_num = data_batch['stroke_number'].to(self.device)
                        predicted_classes = data_batch['category'].to(self.device)
                        self.optim.zero_grad()

                        seg_outs = self.model(strokes=strokes,
                                              stroke_locations=stroke_locations,
                                              sketch_stroke_num=sketch_stroke_num,
                                              sketch_category=predicted_classes)

                        loss = self.calculate_loss(data_batch, seg_outs)
                        model_outputs.loss += loss.item()
                        loss.backward()
                        self.optim.step()
                        model_outputs, _, _ = self.get_outputs(data_batch, seg_outs, model_outputs)

                    break

                except RuntimeError as e:
                    if "DataLoader timed out" in str(e):
                        print("DataLoader timeout occurred. Retrying...")
                    else:
                        raise

            model_outputs.loss /= model_outputs.total_number_of_sketches
            if self.opt['experiment_logging']:
                self.metric_reporter.report_graphs(epoch_id, 'Training', model_outputs)
            self.scheduler.step()

            if epoch_id % self.iter_test == 0:
                valid_model_outputs = self.test()
                f1_dict = self.metric_reporter.calculate_f1_scores(valid_model_outputs)
                f1 = f1_dict['segmentation_f1']
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    print(f'Stroke segmentation f1 score: {f1}')
                    self.metric_reporter.calculate_segmentation_accuracy(valid_model_outputs)
                    if self.save_checkpoint:
                        save_checkpoint(self.model, 'best_model.pth', self.opt, self.segmentation_labels)

                if self.opt['experiment_logging']:
                    self.metric_reporter.report_graphs(epoch_id, 'Validation', valid_model_outputs)
                    self.metric_reporter.report_f1_scores(epoch_id, 'Validation', valid_model_outputs)
                    self.metric_reporter.report_plots(epoch_id, 'Validation', valid_model_outputs)

    def validate(self) -> None:

        model_outputs = self.test()
        self.metric_reporter.calculate_segmentation_accuracy(model_outputs)
        f1_scores = self.metric_reporter.calculate_f1_scores(model_outputs)

        print(f'Segmentation f1 score: {f1_scores["segmentation_f1"]}')
        print(f'Segmentation accuracy: {model_outputs.get_segmentation_accuracy()}')
        if self.opt['experiment_logging']:
            self.metric_reporter.report_plots(0, 'Validation', model_outputs)


class ClusteringTrainer:

    def __init__(self, train_loader, test_loader, model, options, cluster_opt):

        self.opt = options
        self.device = torch.device(f"cuda:{options['gpu']}" if options['device'] == "gpu" else "cpu")
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optim = torch.optim.Adam(model.parameters(), cluster_opt['learning_rate'],
                                      weight_decay=cluster_opt['weight_decay'])
        self.scheduler = CosineAnnealingWarmRestarts(self.optim, T_0=100)
        self.doodle_meta = DoodleDatasetMeta()
        self.max_epoch = self.opt['max_epochs']
        self.best_score = 0
        self.iter_test = 3
        self.save_checkpoint = True

        self.triplet_loss = TripletLoss(margin=1.0)
        self.kmeans = KMeans(n_clusters=2, random_state=42)
        self.metric_reporter = MetricsReporter(metrics_to_report=["loss", "segmentation_accuracy"])

        if self.save_checkpoint:
            self.file_paths = FilePaths(self.opt)
            self.file_paths.check_all()

    def calculate_metrics(self, embeddings, sketch_stroke_num, leg_ids):

        local_sketch_num = 0
        local_stroke_num = 0
        local_correct_strokes = 0

        for i in range(len(sketch_stroke_num)):
            sketch_embeddings = embeddings[i]
            num_strokes = sketch_stroke_num[i].item()
            valid_embeddings = sketch_embeddings[:num_strokes].detach().cpu().numpy()

            if len(valid_embeddings) < 2:
                continue

            cluster_ids = self.kmeans.fit_predict(valid_embeddings)
            labels = leg_ids[i][:num_strokes].detach().cpu().numpy()

            num_correct = max(
                (cluster_ids == labels).sum(),
                (cluster_ids == 1 - labels).sum()
            )

            local_correct_strokes += num_correct
            local_stroke_num += num_strokes
            local_sketch_num += 1

        return local_sketch_num, local_stroke_num, local_correct_strokes

    def test(self) -> ModelOutputs:

        model_outputs = ModelOutputs()
        with torch.no_grad():
            self.model.eval()
            while True:
                try:
                    for data_batch in self.test_loader:

                        self.model.eval()
                        strokes = data_batch['strokes'].to(self.device)
                        sketch_stroke_num = data_batch['stroke_number'].to(self.device)
                        leg_ids = data_batch['seg_label']
                        stroke_ind = data_batch['stroke_labels'].to(self.device)

                        embeddings = self.model(strokes,
                                                sketch_stroke_num,
                                                stroke_class=stroke_ind)

                        anchors, positives, negatives = self.triplet_loss.sample_triplets(embeddings,
                                                                                          leg_ids,
                                                                                          sketch_stroke_num)
                        if anchors is None: continue
                        loss = self.triplet_loss(anchors, positives, negatives)
                        model_outputs.loss += loss.item()

                        delta_sketch_num, delta_stroke_num, delta_correct_strokes = self.calculate_metrics(embeddings,
                                                                                                           sketch_stroke_num,
                                                                                                           leg_ids)
                        model_outputs.total_number_of_sketches += delta_sketch_num
                        model_outputs.total_number_of_strokes += delta_stroke_num
                        model_outputs.correct_segmentation += delta_correct_strokes

                    break

                except RuntimeError as e:
                    if "DataLoader timed out" in str(e):
                        print("DataLoader timeout occurred. Retrying...")
                    else:
                        raise

        model_outputs.loss /= model_outputs.total_number_of_sketches
        return model_outputs

    def train(self) -> None:

        for epoch_id in range(self.max_epoch):
            while True:
                try:
                    print(f"Epoch: {epoch_id + 1}")
                    model_outputs = ModelOutputs()

                    for data_batch in tqdm(self.train_loader):

                        self.model.train()
                        strokes = data_batch['strokes'].to(self.device)
                        sketch_stroke_num = data_batch['stroke_number'].to(self.device)
                        leg_ids = data_batch['seg_label']
                        stroke_ind = data_batch['stroke_labels'].to(self.device)

                        self.optim.zero_grad()
                        embeddings = self.model(strokes, sketch_stroke_num, stroke_class=stroke_ind)

                        anchors, positives, negatives = self.triplet_loss.sample_triplets(embeddings, leg_ids,
                                                                                          sketch_stroke_num)
                        if anchors is None:
                            continue

                        loss = self.triplet_loss(anchors, positives, negatives)
                        model_outputs.loss += loss.item()
                        loss.backward()
                        self.optim.step()

                        delta_sketch_num, delta_stroke_num, delta_correct_strokes = self.calculate_metrics(embeddings,
                                                                                                           sketch_stroke_num,
                                                                                                           leg_ids)
                        model_outputs.total_number_of_sketches += delta_sketch_num
                        model_outputs.total_number_of_strokes += delta_stroke_num
                        model_outputs.correct_segmentation += delta_correct_strokes

                    break

                except RuntimeError as e:
                    if "DataLoader timed out" in str(e):
                        print("DataLoader timeout occurred. Retrying...")
                    else:
                        raise

            model_outputs.loss /= model_outputs.total_number_of_sketches
            if self.opt['experiment_logging']:
                self.metric_reporter.report_graphs(epoch_id, 'Training', model_outputs)
            self.scheduler.step()

            if epoch_id % self.iter_test == 0:
                valid_model_outputs = self.test()
                valid_accuracy = valid_model_outputs.get_segmentation_accuracy()
                if valid_accuracy > self.best_score:
                    self.best_score = valid_accuracy
                    print(f'Accuracy: {valid_accuracy}')
                    if self.save_checkpoint:
                        save_checkpoint(self.model, 'best_model.pth', self.opt)

                if self.opt['experiment_logging']:
                    self.metric_reporter.report_graphs(epoch_id, 'Validation', valid_model_outputs)

    def validate(self) -> None:
        valid_model_outputs = self.test()
        valid_accuracy = valid_model_outputs.get_segmentation_accuracy()
        print(f'Accuracy: {valid_accuracy}')

        if self.opt['experiment_logging']:
            self.metric_reporter.report_graphs(0, 'Validation', valid_model_outputs)
