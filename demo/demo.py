# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import random
from collections import Counter

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


# Example arguments:
# --config-file configs/Experiments/mask_rcnn_R_50_FPN_1x_cls_agnostic_base.yaml
# --input datasets/coco/val2014/COCO_val2014_000000548126.jpg
# --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn_R_50_FPN_cls_ag_b/model_final.pth

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=3,
        help="The amount of random samples to sample from the cfg.DATASETS.TEST dataset in order to perform"
             "inference on. If 10000 then all dataset is shown"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    gt_detections_per_img = None
    gt_category_counter_per_img = None
    # Populate args.input either with manually selected files or with random samples from the dataset
    if args.input:  # Manually selected files
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
    else:  # Randomly chosen files from the dataset
        dataset_dict = DatasetCatalog.get(cfg.DATASETS.TEST[0])
        metadata_dict = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        if args.random_samples == 10000:
            random_dataset_sample = [f for f in dataset_dict]
        else:
            random_dataset_sample = [f for f in random.sample(dataset_dict, args.random_samples)]

        gt_detections_per_img = [[metadata_dict.thing_classes[anno['category_id']]
                                 for anno in sample['annotations'] if anno['iscrowd'] == 0]
                                 for sample in random_dataset_sample
                                 ]
        gt_category_counter_per_img = [Counter(annos) for annos in gt_detections_per_img]


        args.input = [os.path.expanduser(f['file_name']) for f in random_dataset_sample]

    for idx, path in enumerate(tqdm.tqdm(args.input, disable=not args.output)):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        if not gt_detections_per_img:
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
        else:
            logger.info(
                "{}: {} out of {} total GT's of {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    sum(gt_category_counter_per_img[idx].values()),
                    gt_category_counter_per_img[idx],
                    time.time() - start_time,
                )
            )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit
