# Important to keep cv2 top import
import cv2
import os
import copy
import json
from collections import defaultdict
import numpy as np
import logging
import torch
import torchvision

from detectron2.data.dataset_mapper import SimpleDatasetMapper
import detectron2.utils
from detectron2.utils import comm
import detectron2.data.detection_utils
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_argument_parser, launch, default_setup, hooks
from detectron2.config import get_cfg, set_global_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, \
    build_detection_test_loader, build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, verify_results, PascalVOCDetectionEvaluator, \
    LVISEvaluator, DatasetEvaluators
from detectron2.data.datasets import register_coco_instances

import OurPaper.training
from OurPaper.dataset import custom_register_dataset, register_test_class_coco
import dutils.imageutils as diu
import dutils.learnutils as dlu
import dutils.simpleutils as dsu
import dutils.detectronutils as ddu


# register_test_class_coco('TestSmallCOCO', [3, 5])
# register_test_class_coco('TestSmallCOCO_test', [10], 'cocosplit/datasplit/5k.json', 'coco/val2014')


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        metadata = MetadataCatalog.get(dataset_name)
        # Note(): Check original meta classes, also tells us if we have a meta dataset
        # Ugly but easier to support than other options
        classes_to_eval = []
        if hasattr(metadata, 'classes_to_eval'):
            classes_to_eval = metadata.classes_to_eval
            print(f'Using meta-dataset with classes {classes_to_eval}')

        evaluator_type = metadata.evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(
                COCOEvaluator(dataset_name, cfg, True, output_folder, classes_to_eval=classes_to_eval))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)

    # Setup loggers
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="doublefewshot")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name='OurPaper')
    return cfg


def main(args):
    cfg = setup(args)

    if args.print_only:
        print(cfg)
        return

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)

        return res

    if 'pure-metric' in args.method:
        OurPaper.training.metric_training(cfg, args)
        return

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


def _check_and_sanitize_args(args):
    """Perform checks on parsed arguments"""
    if args.method == 'pure-metric-finetuned':
        assert args.src1

    return args


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args = _check_and_sanitize_args(args)

    print('Executing training script!')
    print("Command Line Args:", args)


    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
