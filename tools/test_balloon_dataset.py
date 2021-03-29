import json
import os
import cv2
import numpy as np
import dutils.imageutils as diu
import dutils.learnutils as dlu
import torch, torchvision
import dutils.simpleutils  as dsu
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_argument_parser, launch, default_setup, hooks
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, \
    build_detection_test_loader, build_detection_train_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator, DatasetEvaluator, verify_results, \
    PascalVOCDetectionEvaluator, DatasetEvaluators
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser
from detectron2.checkpoint import DetectionCheckpointer

cfg = get_cfg()
cfg.OUTPUT_DIR = './output/balloon_output/'


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
        # Check original meta classes, also tells us if we have a meta dataset
        # Ugly but easier to support than other options
        classes_to_eval = None
        if hasattr(metadata, 'classes_to_eval'):
            classes_to_eval = metadata.classes_to_eval
            print(f'Using meta-dataset with classes {classes_to_eval}')

        #TODO(): Temp hack. Force COCO evaluation for the moment on all datasets.
        evaluator_list.append(
            COCOEvaluator(dataset_name, cfg, True, output_folder, classes_to_eval=classes_to_eval))
        return evaluator_list[0]

        evaluator_type = metadata.evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(
                COCOEvaluator(dataset_name, cfg, True, output_folder, classes_to_eval=classes_to_eval))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


from detectron2.data import DatasetCatalog, MetadataCatalog

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("datasets/balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"], evaluator_type="coco")
balloon_metadata = MetadataCatalog.get("balloon_train")

def setup_balloon(args):
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file('Experiments/mask_rcnn_R_50_FPN_1x_cls_agnostic_base.yaml'))

    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ('balloon_val',)
    cfg.TEST.EVAL_PERIOD = 300
    cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        # "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join('checkpoints/coco/faster_rcnn_R_50_FPN_cls_ag_b', "model_final.pth")
    cfg.MODEL.WEIGHTS = os.path.join('checkpoints/coco/faster_rcnn_R_50_FPN_cls_ag_b', "modified_model.pth")

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
    # cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
    cfg.MODEL.ROI_MASK_HEAD.FREEZE = True
    """
    Create configs and perform basic setups.
    """
    # cfg = get_cfg()
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="BaloonModel")
    return cfg


def setup_pascal_voc(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('PascalVOC-Detection/faster_rcnn_R_50_C4.yaml'))
    cfg.DATASETS.TEST = ('voc_2007_test_2012_eval',)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
         "PascalVOC-Detection/faster_rcnn_R_50_C4.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="PascalVOCModel")
    return cfg

def main(args):
    # cfg = setup_balloon(args)
    cfg = setup_pascal_voc(args)

    if args.eval_only: 
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)

        # TODO(): Rewrite this
        # if cfg.TEST.AUG.ENABLED:
        #     res.update(Trainer.test_with_TTA(cfg, model))
        return res
    """
       If you'd like to do anything fancier than the standard training logic,
       consider writing your own training loop or subclassing the trainer.
       """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    # if cfg.TEST.AUG.ENABLED:
    #     trainer.register_hooks(
    #         [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
    #     )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.num_gpus = 1
    args.eval_only = True
    args.resume = False

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
