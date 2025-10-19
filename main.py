import torch
import torch.utils.data
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.dataloader import default_collate
from clearml import Task

import opts
from dataset.doodle_dataset import *
from dataset.doodle_dataset_utils import split_dataset, split_dataset_xml
from models.clustering import Leg_Encoder
from models.sketch_transformer import SketchSegmentator
from models.trainers import *
from utils.log_utils import *


torch.multiprocessing.set_sharing_strategy('file_system')

train_transforms = d.TransformsCompose([
    d.StrokeDropout(dropout_num=2, min_sketch_strokes=3, p=0.5, smallest=True),
    d.RandomScaleOR((0.8, 1.2), (0.8, 1.2), p=0.9),
    d.RandomRotate(-30.0, 30.0, p=0.9),
    d.RandomVerticalFlip(p=0.5),
    d.RandomStrokeRotate((-7.0, 7.0), stroke_num=2, p=0.9),
    d.RandomStrokeScaleOR((0.9, 1.1), (0.9, 1.1), stroke_num=3, p=0.5),
    d.RandomStrokeShift((-3, 3), (-3, 3), stroke_num=3, p=0.5),
    d.GaussianNoise(window_size=3, p=0.3)
])
test_transforms = None


def report_hyperparameters(report, options, name):
    if report:
        task = Task.current_task()
        task.connect(options, name=name)


def custom_collate_fn(batch):
    flattened_batch = [item for sublist in batch for item in sublist]
    return default_collate(flattened_batch)


def get_dataset(dataset_types: List, opt: Dict[str, Any] = None, transforms: TransformsCompose = None,
                mode: str = None, multiplication_factor: int = None, model_type='segmentator') -> Any:
    datasets = []
    for d_type in dataset_types:
        if d_type == 'xml':
            dataset_xml = DoodleDatasetXML(data_dir=opt['dataset_path_xml'], split_file_name=opt['data_split_xml'],
                                           mode=mode, transform=transforms, model_type=model_type)
            datasets.append(dataset_xml)
        elif d_type == 'json':
            dataset_json = DoodleDatasetJSON(data_dir=opt['dataset_path'], split_file_name=opt['data_split_json'],
                                             mode=mode, transform=transforms, model_type=model_type)
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


def load_model(model_class, model_path, device, options=None):
    print(f'Loading pretrained model from {model_path}')
    model = model_class(device, options).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model


def load_pretrained_model_by_type(model_type, meta, device, options_dict, other_opt):

    model_paths = {
        'segmentator': 'experiments/10-3-17-24-14/checkpoint/best_model.pth',
        'clustering': 'experiments/10-3-16-8-51/checkpoint/best_model.pth',
    }

    model_classes = {
        'clustering': Leg_Encoder,
        'segmentator': SketchSegmentator
    }

    model_path = model_paths[model_type]
    other_opt[f'{model_type}_path'] = model_path

    if model_type == 'clustering':
        options_dict['image_size'] = meta.CLUSTERING_IMAGE_SIZE
        options_dict['max_stroke'] = meta.MAX_STROKE_NUM

    else:
        options_dict['image_size'] = meta.IMAGE_SIZE
        options_dict['num_labels'] = meta.NUM_SKETCH_CLASSES
        options_dict['num_segments'] = meta.NUM_STROKE_LABELS
        options_dict['max_stroke'] = meta.MAX_STROKE_NUM

    return load_model(model_classes[model_type], model_path, device, options_dict)


def initialize_model_and_trainer(model_type, meta, device, train_loader, test_loader, opt, segmentator_opt, cluster_opt,
                                 other_opt):
    if model_type == 'clustering':
        cluster_opt['image_size'] = meta.CLUSTERING_IMAGE_SIZE
        cluster_opt['max_stroke'] = meta.MAX_STROKE_NUM
        model = Leg_Encoder(device, cluster_opt).to(device)
        trainer = ClusteringTrainer(train_loader, test_loader, model, opt, cluster_opt)
        report_hyperparameters(opt['experiment_logging'], cluster_opt, "Clustering model parameters")

    else:
        if segmentator_opt['use_pretrained']:
            model = load_pretrained_model_by_type('segmentator', meta, device, segmentator_opt, other_opt)
        else:
            segmentator_opt['image_size'] = meta.IMAGE_SIZE
            segmentator_opt['num_labels'] = meta.NUM_SKETCH_CLASSES
            segmentator_opt['num_segments'] = meta.NUM_STROKE_LABELS
            segmentator_opt['max_stroke'] = meta.MAX_STROKE_NUM
            model = SketchSegmentator(device, segmentator_opt).to(device)

        clustering_model = None
        if segmentator_opt.get('use_leg_clustering'):
            clustering_model = load_pretrained_model_by_type('clustering', meta, device, cluster_opt, other_opt)
            report_hyperparameters(opt['experiment_logging'], cluster_opt, "Clustering model parameters")

        trainer = SegmentatorTrainer(train_loader, test_loader, model, opt, segmentator_opt,
                                     clustering_model=clustering_model)
        report_hyperparameters(opt['experiment_logging'], segmentator_opt, "Segmentator parameters")

    if trainer and trainer.save_checkpoint:
        other_opt['trained_model_path'] = opt['experiment_folder'] + opt['ckpt_folder'] + 'best_model.pth'

    return model, trainer


def main(opt, segmentator_opt, cluster_opt):

    other_opt = {}
    batch_size = opt['bs']
    gpu_index = opt['gpu']
    device = torch.device(f"cuda:{gpu_index}" if opt['device'] == "gpu" else "cpu")
    print(f"Running on {device}")
    set_seed()

    proportions = {'train': 0.8, 'test': 0.2}

    split_dataset(proportions, opt['data_split_json'], opt['dataset_path'])
    split_dataset_xml(proportions, opt['data_split_xml'], opt['dataset_path_xml'])

    model_type = 'segmentator'  # 'segmentator' or 'clustering'
    doodle_meta = DoodleDatasetMeta()

    train_set = get_dataset(['json', 'xml'], opt=opt, mode='train', transforms=train_transforms,
                            multiplication_factor=3, model_type=model_type)
    test_set = get_dataset(['json', 'xml'], opt=opt, mode='test', transforms=test_transforms, model_type=model_type)

    if model_type == 'clustering':
        batch_size = int(batch_size / 2)
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16, shuffle=True, timeout=30,
                                  collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=1, shuffle=False, timeout=30,
                                 collate_fn=custom_collate_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=20, shuffle=True, timeout=30)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4, shuffle=False, timeout=30)

    # class_counter, stroke_counter, _, _ = compute_dataset_statistics(train_loader, doodle_meta, doodle_meta.class_multipliers)
    # _, _, _, _ = compute_dataset_statistics(test_loader, doodle_meta)
    other_opt['model_type'] = model_type
    other_opt['oversampling_factors'] = json.dumps(dict(doodle_meta.class_multipliers), indent=2)
    # other_opt['class_counter'] = json.dumps(dict(class_counter), indent=2)
    # other_opt['stroke_counter'] = json.dumps(dict(stroke_counter), indent=2)
    other_opt['augmentations'] = [str(t) for t in train_transforms.transforms]
    other_opt['loss'] = opt['loss_type']

    model, trainer = initialize_model_and_trainer(model_type,
                                                  doodle_meta,
                                                  device,
                                                  train_loader,
                                                  test_loader,
                                                  opt,
                                                  segmentator_opt,
                                                  cluster_opt,
                                                  other_opt)

    display_experiment_params(model, train_transforms, train_set, opt)
    report_hyperparameters(opt['experiment_logging'], other_opt, 'Other options')
    trainer.train()
    #trainer.validate()


if __name__ == "__main__":
    opt = opts.parse_general_args()
    opt = vars(opt)
    segmentator_opt = opts.parse_segmentator_opt()
    segmentator_opt = vars(segmentator_opt)
    cluster_opt = opts.parse_cluster_opt()
    cluster_opt = vars(cluster_opt)
    log_experiment(opt['experiment_logging'])
    main(opt, segmentator_opt, cluster_opt)
