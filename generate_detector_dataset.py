import opts
from detector_pipeline.detector_dataset import *
from detector_pipeline.detector_pipeline_utils import *
from utils.log_utils import *
from dataset.doodle_dataset_utils import split_dataset, split_dataset_xml


transforms = d.TransformsCompose([
    d.RandomScaleOR((0.9, 1.1), (0.9, 1.1), p=0.9),
    d.RandomRotate(-20.0, 20.0, p=0.9),
    d.RandomVerticalFlip(p=0.5)
])


if __name__ == "__main__":

    opt = opts.parse_general_args()
    opt = vars(opt)
    set_seed()

    proportions = {'train': 0.8, 'test': 0.2}
    class_overrides = {
        'cloud': {'train': 0.6, 'test': 0.4},
        '2leg': {'train': 0.6, 'test': 0.4},
        'flower': {'train': 0.7, 'test': 0.3},
    }
    split_dataset(proportions, opt['data_split_json'], opt['dataset_path'], class_overrides)
    split_dataset_xml(proportions, opt['data_split_xml'], opt['dataset_path_xml'], class_overrides)

    doodle_meta = DoodleDatasetDetectorMeta()

    train_set = get_detector_dataset(['json', 'xml'], opt=opt, mode='train', multiplication_factor=3, transforms=transforms)
    test_set = get_detector_dataset(['json', 'xml'], opt=opt, mode='test', multiplication_factor=1, transforms=None)

    create_dataset(train_set, out_dir="yolo_dataset", instances=18000, split="train")
    print("Generated train set!")
    create_dataset(test_set, out_dir="yolo_dataset", instances=2000, split="val")
    print("Generated test set!")

    create_data_yaml(dataset_root="yolo_dataset2", num_classes=7, class_names=["car", "cloud", "flower", "sun", "tree", "4leg", "2leg"])


