"""
Computes a CSV from the aggregated results for a bunch of seeds, and for one 'shot' type.

Once results have been aggregated using aggregate_seeds.py they are placed in an
'aggregate-folder'. This changes based on the dataset in either 'aggregate-folder/coco'
or 

We then take every folder in this aggregate-folder directory, and if it is of the same type and of the same number of
shots, we add it to a 'csv' file.

This way we can easily generate CSV files/tables with our results.

Usage: python --{coco} --type novel --shot 5 --aggregate-folder 'checkpoints/aggregated_results'
"""
import argparse
import csv
import json
import os


def parse_args(given_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['all', 'novel'], type=str, required=True)
    parser.add_argument('--shot', type=int, required=True)
    # Folder where the aggregation results from aggregate_seeds.py is.
    parser.add_argument('--aggregate-folder', type=str, default='checkpoints/aggregated_results/')

    parser.add_argument('--coco', action='store_true')

    parser.add_argument('--output-confidence-interval', action='store_true')

    args = parser.parse_args(given_args)

    return args


if __name__ == '__main__':
    args = parse_args()
    # args =  parse_args('--coco --type novel --shot 5'.split())

    # args.type = 'novel'
    # args.shot = 5
    # args.coco = True
    # args.type = 'novel'
    # args.shot = 5
    # args.aggregate_folder = 'checkpoints/aggregated_results'

    # Require at least one to be set but not both. Use XOR
    if not args.coco:
        assert False

    data_str = 'coco' if args.coco else None
    if not data_str :
        raise ValueError("give coco as arg")
    aggregate_root_folder = args.aggregate_folder + '/' + data_str

    tracked_metrics = []
    if args.type == 'novel':
        tracked_metrics = ['nAP', 'nAP50', 'nAP75', 'nAPs', 'nAPm', 'nAPl']
    elif args.type == 'all':
        tracked_metrics = ['AP', 'AP50', 'bAP', 'bAP50', 'nAP', 'nAP50']

    tracked_metrics_per_type = ['bbox-' + metric for metric in tracked_metrics]
    tracked_metrics_per_type = tracked_metrics_per_type + ['segm-' + metric for metric in
                                                           tracked_metrics]
    fieldnames = ['name'] + tracked_metrics_per_type

    csv_output_folder = os.path.join(aggregate_root_folder, 'csvs')
    csv_file_name = f'{args.shot}{args.type}.csv'
    csv_full_path = os.path.join(csv_output_folder, csv_file_name)
    os.makedirs(csv_output_folder, exist_ok=True)

    with open(csv_full_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for path in os.listdir(aggregate_root_folder):
            try:
                if f'{args.shot}shot' in path and args.type in path:
                    result_dict = {'name': path.split('mask_rcnn_')[1]}

                    simplestats_json_path = os.path.join(aggregate_root_folder, path,
                                                        'aggregated_statistics.json')
                    with open(simplestats_json_path, 'r') as j:
                        loaded_json = json.load(j)
                    for res_type in ['bbox', 'segm']:
                        for tracked_metric in tracked_metrics:
                            output_str = f"{loaded_json[res_type][tracked_metric]['mean']:.2f}"
                            if args.output_confidence_interval:
                                output_str += " \"+-\" " \
                                            f"{loaded_json[res_type][tracked_metric]['conf_95']:.2f}"

                            result_dict[
                                res_type + '-' + tracked_metric] = output_str
                    writer.writerow(result_dict)
            except KeyError as ke:
                print(ke)
                print(f'Key error at {path}')

    print(f"Results written to {csv_full_path}")
