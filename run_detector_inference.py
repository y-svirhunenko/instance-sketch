import torch
import torch.utils.data
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.dataloader import default_collate
from clearml import Task

import opts

from dataset.doodle_dataset_utils import split_dataset, split_dataset_xml
from detector_pipeline.detector_dataset import *
from detector_pipeline.detector_pipeline_utils import *
from models.clustering import Leg_Encoder
from models.sketch_transformer import SketchSegmentator
from models.trainers import *
from utils.log_utils import *


transforms = d.TransformsCompose([
    d.RandomScaleOR((0.9, 1.1), (0.9, 1.1), p=0.9),
    d.RandomRotate(-20.0, 20.0, p=0.9),
    d.RandomVerticalFlip(p=0.5)
])

def report_hyperparameters(report, options, name):
    if report:
        task = Task.current_task()
        task.connect(options, name=name)


def load_model(model_class, model_path, device, options=None):
    print(f'Loading pretrained model from {model_path}')
    model = model_class(device, options).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model


def load_pretrained_model_by_type(model_type, meta, device, options_dict, other_opt):

    model_paths = {
        'clustering': 'experiments/10-3-16-8-51/checkpoint/best_model.pth',
        'segmentator': 'experiments/10-3-17-24-14/checkpoint/best_model.pth'
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


def main(opt, segmentator_opt, cluster_opt):

    other_opt = {}
    gpu_index = opt['gpu']
    device = torch.device(f"cuda:{gpu_index}" if opt['device'] == "gpu" else "cpu")
    print(f"Running on {device}")
    set_seed()

    proportions = {'train': 0.8, 'test': 0.2}

    split_dataset(proportions, opt['data_split_json'], opt['dataset_path'])
    split_dataset_xml(proportions, opt['data_split_xml'], opt['dataset_path_xml'])

    doodle_meta = DoodleDatasetDetectorMeta()
    set = get_detector_dataset(['json', 'xml'], opt=opt, mode='test', multiplication_factor=1, transforms=None)
    loader = DataLoader(set, batch_size=1, num_workers=1, shuffle=False)  ##BS = 1

    clustering_model = load_pretrained_model_by_type('clustering', doodle_meta, device, cluster_opt, other_opt)
    segmentator = load_pretrained_model_by_type('segmentator', doodle_meta, device, segmentator_opt, other_opt)
    detector_tester = DetectorPipelineTester(loader, other_opt, device, segmentator, clustering_model,
                                             experiment_logging=opt['experiment_logging'])

    detector_tester.validate()


if __name__ == "__main__":
    opt = opts.parse_general_args()
    opt = vars(opt)
    segmentator_opt = opts.parse_segmentator_opt()
    segmentator_opt = vars(segmentator_opt)
    cluster_opt = opts.parse_cluster_opt()
    cluster_opt = vars(cluster_opt)
    log_experiment(opt['experiment_logging'])
    main(opt, segmentator_opt, cluster_opt)
