import argparse
import time


def parse_general_args():

    parser = argparse.ArgumentParser()
    t = time.localtime()

    parser.add_argument(
        '--home', '-home',
        type=str,
        default='./')

    parser.add_argument(
        '--experiment_folder', '-experiment_folder',
        type=str,
        default='experiments/' + str(t.tm_mon) + "-" + str(t.tm_mday) + "-" + str(t.tm_hour) + "-" + str(
            t.tm_min) + "-" + str(t.tm_sec) + "/")

    parser.add_argument(
        '--ckpt_folder', '-ckpt_folder',
        type=str,
        default='checkpoint/')

    parser.add_argument(
        '--log_folder', '-log_folder',
        type=str,
        default='log/')

    parser.add_argument(
        '--model_outputs_folder', '-model_outputs_folder',
        type=str,
        default='model_outputs/')

    parser.add_argument(
        '--log_file', '-log_file',
        type=str,
        default='best_metrics.txt',
        help='Path to optuna outputs'
    )

    parser.add_argument(
        '--raw_outputs_file', '-raw_outputs_file',
        type=str,
        default='outputs.ndjson',
        help='Path to optuna outputs'
    )

    parser.add_argument(
        '--dataset_path', '-dataset_path',
        type=str,
        default='data/segmentation',
        help='Path to the dataset folder'
    )

    parser.add_argument(
        '--dataset_path_xml', '-dataset_path_xml',
        type=str,
        default='data/annotations_xml',
        help='Path to the dataset folder'
    )

    parser.add_argument(
        '--data_split_json', '-data_split_json',
        type=str,
        default='splits/split.json',
        help='Path to the json that records dataset splits'
    )

    parser.add_argument(
        '--data_split_xml', '-data_split_xml',
        type=str,
        default='splits/split_xml.json',
        help='Path to the json that records dataset splits'
    )

    parser.add_argument(
        '--loss_type', '-loss_type',
        type=str,
        default='ce',
        help='The loss to use for classification tasks'
    )


    parser.add_argument(
        '--bs', '-bs',
        type=int,
        default=128,
        help='Batch size for training'
    )

    parser.add_argument(
        '--gpu', '-gpu',
        type=int,
        default=0,
        help='GPU index for training')

    parser.add_argument(
        "--max_epochs", '-max_epochs',
        type=int,
        default=150,
        help="Max count of training epochs"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Whether to use cpu or gpu"
    )

    parser.add_argument(
        '--experiment_logging', '-experiment_logging',
        type=_str_to_bool, nargs='?', const=True,
        default=False,
        help='Whether to log experiment'
    )

    args = parser.parse_args()
    return args



def parse_segmentator_opt():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate', '-learning_rate',
        type=float,
        default=3e-4,
        help='Learning rate for Adam optimizer'
    )

    parser.add_argument(
        '--weight_decay', '-weight_decay',
        type=float,
        default=0.0001,
        help='Weight decay for Adam optimizer'
    )

    parser.add_argument(
        '--embedding_dropout', '-embedding_dropout',
        type=float,
        default=0,
        help='Dropout rate for embedding layers'
    )

    parser.add_argument(
        '--attention_dropout', '-attention_dropout',
        type=float,
        default=0,
        help='Dropout rate for attention layers'
    )

    parser.add_argument(
        '--dropout_rate', '-dropout_rate',
        type=float,
        default=0,
        help='Dropout rate'
    )

    parser.add_argument(
        '--hidden_size', '-hidden_size',
        type=int,
        default=128,
        help='Hidden size'
    )

    parser.add_argument(
        '--encoder_layers', '-encoder_layers',
        type=int,
        default=2,
        help='Number of encoder layers'
    )

    parser.add_argument(
        '--encoder_attention_heads', '-encoder_attention_heads',
        type=int,
        default=4,
        help='Number of encoder attention_heads'
    )

    parser.add_argument(
        '--use_leg_clustering', '-use_leg_clustering',
        type=_str_to_bool, nargs='?', const=True,
        default=False,
        help='Whether to use leg clustering'
    )

    parser.add_argument(
        '--use_pretrained', '-use_pretrained',
        type=_str_to_bool, nargs='?', const=True,
        default=False,
        help='Whether to use pretrained model'
    )

    args = parser.parse_args()
    return args


def parse_cluster_opt():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate', '-learning_rate',
        type=float,
        default=3e-4,
        help='Learning rate for Adam optimizer'
    )

    parser.add_argument(
        '--weight_decay', '-weight_decay',
        type=float,
        default=0.00001,
        help='Weight decay for Adam optimizer'
    )

    parser.add_argument(
        '--hidden_size', '-hidden_size',
        type=int,
        default=64,
        help='Hidden size'
    )

    parser.add_argument(
        '--embedding_size', '-embedding_size',
        type=int,
        default=32,
        help='Embedding size'
    )

    parser.add_argument(
        '--embedding_dropout', '-embedding_dropout',
        type=float,
        default=0,
        help='Dropout rate for embedding layers'
    )

    parser.add_argument(
        '--attention_dropout', '-attention_dropout',
        type=float,
        default=0,
        help='Dropout rate for attention layers'
    )

    parser.add_argument(
        '--dropout_rate', '-dropout_rate',
        type=float,
        default=0,
        help='Dropout rate'
    )

    parser.add_argument(
        '--encoder_layers', '-encoder_layers',
        type=int,
        default=2,
        help='Number of encoder layers'
    )

    parser.add_argument(
        '--encoder_attention_heads', '-encoder_attention_heads',
        type=int,
        default=4,
        help='Number of encoder attention_heads'
    )

    args = parser.parse_args()
    return args


def _str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
