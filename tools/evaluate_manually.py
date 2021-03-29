"""
    Run the COCO eval script on a manual instance of instance_predictions.pth

    In the case of testing on an RPN and with region proposals, this instance_predictions.pth
    is created by running an --eval-only script in --run-train for the RPN model.
"""
import cv2

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path
import json

import detectron2.data.detection_utils
import detectron2.utils
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.data.dataset_mapper import SimpleDatasetMapper
from detectron2.evaluation import (
    COCOEvaluator,
)
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

import dutils.detectronutils as ddu
import dutils.imageutils as diu
import dutils.learnutils as dlu
import dutils.simpleutils as dsu


def parse_args(passed_args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions', type=Path,
                        help='Path to predictions (instance_predictions or box predictions)'
                        'from model results')
    parser.add_argument('--eval-result-ext', type=Path, default='inference_revised',
                        help="Extension to output file")
    #TODO(): Allow eval-type
    # parser.add_argument('--eval-type', type=str, choices=['test','base','all'])

    args = parser.parse_args(passed_args)

    return args

def compute_cfg(output_root):
    cfg = get_cfg()
    cfg.merge_from_file(str(output_root / 'config.yaml'))
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.KEYPOINT_ON = False
    cfg.TEST.KEYPOINT_OKS_SIGMAS = False
    return cfg

def main():
    setup_logger()
    # test_instance_pred_path = 'checkpoints/coco/mask_rcnn_R_50_FPN_ft_fullclsag_cos_bh_novel_5shot/inference/instances_predictions.pth'
    test_prediction_path= 'checkpoints/coco/rpn_R_50_FPN_ft_fullclsag_cos_bh_novel_5shot/inference/instances_predictions.pth'
    args = parse_args(test_prediction_path.split())

    output_root = args.predictions.parent.parent

    print(f'Root of model is {output_root}')
    cfg = compute_cfg(output_root)

    # Get the classes we wish to evaluate for.
    test_dataset_name = cfg.DATASETS.TEST[0]
    metadata = MetadataCatalog.get(test_dataset_name)

    print(f'Test Dataset Name is {test_dataset_name}')

    eval_root = output_root / args.eval_result_ext
    evaluator = COCOEvaluator(test_dataset_name, cfg, False, str(eval_root), metadata.classes_to_eval)
    results = evaluator.evaluate_with_predictions(str(args.predictions))

    assert results != None

    with open(eval_root / 'res_final.json', 'w') as fp:
        json.dump(results, fp, indent = 4)

if __name__ == "__main__":
    main()
