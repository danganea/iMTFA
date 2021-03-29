"""
Usage:

    python tools/aggregate_seeds.py --experiment-name ... --coco --shots
    Example:
    python tools/aggregate_seeds.py --experiment-name 
"""
import argparse
import json
import math
import numpy as np
import os
import re
from collections import OrderedDict, defaultdict
from typing import List, Optional
from enum import Enum


class DatasetType(Enum):
    COCO = 1


class AggregationDatasetConfig():

    def __init__(self, experiment_name : str, dataset_type: DatasetType, output_folder: Optional[str] = None,
                 results_folder: Optional[str] = None):
        self.experiment_name = experiment_name
        self.dataset = dataset_type.name.lower()
        self.output_folder = 'checkpoints/aggregated_results/'
        # Handle per-dataset specific options
        if dataset_type == DatasetType.COCO:
            self.result_folder = 'checkpoints/nfsmirror/faster_rcnn/'
        else:
            raise ValueError("Datasets can only be 'coco'")
        
        # Overwrite settings if arguments were passed in
        self.output_folder = output_folder if output_folder else self.output_folder
        self.result_folder = results_folder if results_folder else self.result_folder

        self.experiment_path = os.path.join(self.output_folder, self.dataset, self.experiment_name)

    def get_inference_file_path(self, seed):
        seed = 'seed{}'.format(seed)
        base_folder = os.path.join(self.result_folder, seed, self.experiment_name)
        inference_file = os.path.join(base_folder, 'inference/res_final.json')

        return inference_file


def _parse_args():
    parser = argparse.ArgumentParser()
    # experiment_name -> mask_rcnn_R_50....novel..
    parser.add_argument('--experiment-name', type=str, required=True, help="Name of an experiment\n"
                        "Full path is not required.\n"
                        "Example: 'mask_rcnn_R_50_FPN_fullclsag_metric_avg_cos_bh_novel_5shot'")
    parser.add_argument('--dataset', type=str, choices=['coco'], required='True')

    # Shots to aggregate over
    parser.add_argument('--shots', type=int, nargs='+', help="Shots to aggregate over")
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--output-folder', type=str,
                        help='Folder in which to place the aggregation results. Gets defaulted otherwise')
    parser.add_argument('--result-folder', type=str, help='Folder in which the results for our'
                        'particular dataset are\n'
                        'Example: checkpoints/nfsmirror/faster_rcnn/ \n'
                        'Gets defaulted if not specified')

    args = parser.parse_args()
    args.dataset_type = DatasetType[args.dataset.upper()]

    return args


def _sanitize_args(args):
    # Discard full path if any present
    _, path_tail = os.path.split(args.experiment_name)
    # Discard extension if any present
    experiment_name, _ = os.path.splitext(path_tail)

    args.experiment_name = experiment_name

    return args


def get_stats_for_metric(values_for_metric):
    mean = np.mean(values_for_metric)
    std = np.std(values_for_metric)
    q1 = np.percentile(values_for_metric, 25)
    median = np.percentile(values_for_metric, 50)
    q3 = np.percentile(values_for_metric, 75)
    # Note: These are lowest and highest indexes into the results array, not actual seed numbers.
    lowest_index = int(np.argmin(values_for_metric))
    lowest = float(values_for_metric[lowest_index])
    highest_index = int(np.argmax(values_for_metric))
    highest = float(values_for_metric[highest_index])

    return {'mean': round(mean,2),
            'std': round(std,2),
            'conf_95': round(1.96 * std / math.sqrt(10),2),
            'Q1': q1,
            'median': median,
            'Q3': q3,
            'lowest': lowest,
            'lowest_index': lowest_index,
            'highest': highest,
            'highest_index': highest_index}


def get_all_experiment_metrics(inference_file: str, metrics: dict) -> Optional[dict]:
    """Gets all the metrics in the inference file to be placed in the passed 'metrics' dict

    :return: If success, returns the metrics, else returns None
    :rtype: bool
    """
    try:
        with open(inference_file, 'r') as f:
            res = json.load(f)
    except Exception as e:
        print(e)
        return None

    for metric_type in ['bbox', 'segm']:
        for metric in res[metric_type]:
            metrics[metric_type][metric].append(res[metric_type][metric])

    return metrics

def _write_statistics(experiment_path, per_seed_dict, simple_aggregated_stats, aggregated_stats):
    os.makedirs(experiment_path, exist_ok=False)

    per_seed_dict_path = os.path.join(experiment_path, 'per_seed_statistics.json')
    simple_aggregated_path = os.path.join(experiment_path, 'simple_statistics.json')
    aggregated_stats_path = os.path.join(experiment_path, 'aggregated_statistics.json')

    def write_json_to_file(json_dict, path):
        with open(path, 'w') as f:
            json.dump(json_dict, f, indent=4)

    write_json_to_file(per_seed_dict, per_seed_dict_path)
    write_json_to_file(simple_aggregated_stats, simple_aggregated_path)
    write_json_to_file(aggregated_stats, aggregated_stats_path)


def experiment_list_from_shots(experiment_name: str, shots: List[int]) -> List[str]:
    """Computes a list of experiment names based on a list of shots, by replacing the shot name
    in each file

    :param experiment_name: Name of the experiment
    :param shots: List containing all the shots
    :return: A list of all experiment names per shot
    :rtype: List[str]
    """
    experiments = {experiment_name}  # Always include current experiment_name.

    if shots:  # If shots present, add more experiments.
        for shot in shots:
            experiment = re.sub('.shot', '{}shot'.format(shot), experiment_name)
            experiments.add(experiment)
    return experiments

def aggregate_csv(experiment_name, dataset_type, output_folder, result_folder, seeds):
    """Aggregates all the experiments which fit the given options and outputs them to output_folder"""

    dataset_config = AggregationDatasetConfig(experiment_name,
                                              dataset_type,
                                              output_folder,
                                              result_folder)

    result_folder = dataset_config.result_folder

    metrics = {'bbox': defaultdict(list), 'segm': defaultdict(list)}

    found_seeds = []
    for seed_index in range(seeds):
        inference_file = dataset_config.get_inference_file_path(seed_index)
        print(f'Seed : {seed_index}; Results file exists: {os.path.exists(inference_file)}')
        result = get_all_experiment_metrics(inference_file, metrics)
        if result:
            found_seeds.append(seed_index)

    main_metrics = ['bAP', 'bAP50', 'nAP', 'nAP50', 'AP', 'AP50']
    # All statistics for individual seeds in one place
    per_seed_dict = {}
    for seed_index, seed in enumerate(found_seeds):
        seed_string = f'seed{seed}'
        per_seed_dict[seed_string] = {}
        for res_type in ['segm', 'bbox']:
            metric_type = metrics[res_type]

            per_seed_dict[seed_string][res_type] = {}
            for metric in main_metrics:
                if metric in metric_type:
                    per_seed_dict[seed_string][res_type][metric] = metric_type[metric][seed_index]
    print(json.dumps(per_seed_dict, indent=4))

    # Print most important metrics first ( i.e. no class based metrics)
    simple_aggregated_stats = OrderedDict([('bbox', {}), ('segm', {})])
    for res_type in ['segm', 'bbox']:
        metric_type = metrics[res_type]
        for metric in metric_type:
            if metric not in main_metrics:
                continue
            simple_aggregated_stats[res_type][metric] = get_stats_for_metric(metric_type[metric])

    print(json.dumps(simple_aggregated_stats, indent=4))

    # Long form statistics. Including all AP variants and per-class metrics
    aggregated_stats = OrderedDict([('bbox', {}), ('segm', {})])
    for res_type in ['segm', 'bbox']:
        metric_type = metrics[res_type]
        for metric in metric_type:
            aggregated_stats[res_type][metric] = get_stats_for_metric(metric_type[metric])

    print(json.dumps(aggregated_stats, indent=4))

    _write_statistics(dataset_config.experiment_path, per_seed_dict,
                     simple_aggregated_stats, aggregated_stats)

    print(f'Finished aggregation script for experiment name {experiment_name}.\n'
          f'Found {len(found_seeds)} seeds\n'
          f'Results placed in {dataset_config.experiment_path} folder')

def main():
    args = _parse_args()

    args = _sanitize_args(args)

    # This is just as a convenience instead of running the script multiple times.
    experiments = experiment_list_from_shots(experiment_name=args.experiment_name, shots=args.shots)

    for experiment in experiments:
        # Discard extension if any present
        experiment, _ = os.path.splitext(experiment)

        try:
            aggregate_csv(experiment, args.dataset_type, args.output_folder,
                          args.result_folder, args.seeds)
        except Exception as e:
            # Perhaps some files don't exist or something, we should continue aggregating though
            print(f'Could not perform experiment aggregation with {args.experiment_name}')
            print(e)


if __name__ == '__main__':
    main()
