import cv2

import random
from collections import Counter
from typing import List, Dict, Tuple
import copy

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg, CfgNode
from PIL import Image
import torch


def get_category_id_dict_from_dataset(data_dict: List[Dict]) -> Dict:
    """
    Get a dictionary with how many times each category appears in all annotations in a given data_dict
    Args:
        data_dict: Data dictionary as returned by DatasetCatalog.get(...)()

    Returns: A dict of (category : appearances) to count how many times each class appears in the data_dict annotations

    """
    return dict(Counter([annotation['category_id'] for file in data_dict for annotation in file['annotations']]))


def inference_setup_cfg(main_cfg, model_weights, score_thresh=0.5):
    pass


def inference_on_one_img(img, demo, img_transform=lambda x: x):
    """
    Runs inference on one image and returns predictions and vizualized_img

    Args:
        img: Either str of image path or an already loaded image in 'BGR' format.
        demo: Visualizer obtained by calling VisualizationDemo(cfg)
        img_transform: Lambda used to transform the image. For example, one can crop it or convert it to BGR.

    Returns:
        tuple:  predictions, visualized_output. One can visualize the results by displaying visualized_output as an img
    """
    if type(img) is str:  # Allow for loading the image
        img = read_image(img, format="BGR")
    img = img_transform(img)

    predictions, visualized_output = demo.run_on_image(img)
    return predictions, visualized_output

def inference_no_overlay(img, demo, img_transform=lambda x: x):
    """
    Runs inference on one image and returns predictions and vizualized_img

    Args:
        img: Either str of image path or an already loaded image in 'BGR' format.
        demo: Visualizer obtained by calling VisualizationDemo(cfg)
        img_transform: Lambda used to transform the image. For example, one can crop it or convert it to BGR.

    Returns:
        tuple:  predictions, visualized_output. One can visualize the results by displaying visualized_output as an img
    """
    if type(img) is str:  # Allow for loading the image
        img = read_image(img, format="BGR")
    img = img_transform(img)

    instances, viz = demo.get_visualizer_and_pred_instances(img)
    return instances,viz 



def remove_model_weights_ourmethod(model: dict):
    return _remove_model_weights(model)


def remove_model_weights_fsdet(model: dict):
    return _remove_model_weights(model, to_delete=[
        'roi_heads.box_predictor.cls_score',
        'roi_heads.box_predictor.bbox_pred',
        'roi_heads.mask_head.predictor'
    ])


def _remove_model_weights(model: dict, to_delete=None) -> dict:
    """
    Removes certain weights of a given model. The weights to remove are given by the to_delete argument.
    If there is also a bias term, that is deleted as well.
    Args:
        model: Loaded detectron2 model
        to_delete (list): Names of the weights to delete from the model, by default:
                     ['roi_heads.box_predictor.cls_score',
                      'roi_heads.box_predictor.bbox_pred']
    """
    assert isinstance(model, dict)
    assert 'model' in model

    # print("Removing model weights with to_delete = None\n It is recommended to specify the to_delete weights directly, or use remove_model_weights_fsdet etc")

    # to_delete default values written here in order for default args to be immutable.
    if to_delete is None:
        # Heads in the bbox predictor:
        to_delete = ['roi_heads.box_predictor.cls_score',
                     'roi_heads.box_predictor.bbox_pred']

    for param_name in to_delete:
        del model['model'][param_name + '.weight']
        if param_name + '.bias' in model['model']:
            del model['model'][param_name + '.bias']

    return model


def reset_model(model: dict) -> dict:
    """
    Resets detectron2 model 'state'. This means current scheduler, optimizer and MOST IMPORTANTLY iterations are deleted
    or set to zero.

    Allows use of this model as a clean_slate.
    """
    assert isinstance(model, dict)
    assert 'model' in model

    if 'scheduler' in model:
        del model['scheduler']
    if 'optimizer' in model:
        del model['optimizer']
    if 'iteration' in model:
        model['iteration'] = 0

    return model


def reset_model_from_path(model_path: str, save_to: object = None) -> None:
    """
    Resets a Detectron2 model loaded from model_path and saves it to save_to path.
    For information on what resetting is, see 'reset_model' function.
    :param model_path: Path to a Detectron2 model
    :param save_to: Path to where you want to save this model. Intermediate directories must exist.
    """
    temp_model = torch.load(model_path)
    reset_model(temp_model)

    if save_to != None:
        print(f'Saving new model to: {save_to}')
        torch.save(temp_model, save_to)

def get_cfg_default():
    return get_cfg()

def get_cfg_from_file(c_file : str):
    cfg = get_cfg()
    cfg.merge_from_file(c_file)
    return cfg

def build_model_and_cfg(c_file: str, w_file=None, trainer=DefaultTrainer) -> Tuple[torch.nn.Module, CfgNode]:
    """
    :param c_file: Config file
    :param w_file: Weight file. None if we're ok with the one in c_file.
    :param trainer: Pass in own trainer to have proper evaluation scripts etc
    :return model, cfg
    """
    cfg = get_cfg()
    cfg.merge_from_file(c_file)
    if w_file:
        cfg.MODEL.WEIGHTS = w_file

    model = trainer.build_model(cfg)
    if w_file:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    return model, cfg


def build_model_from_cfg(cfg: CfgNode, load_weights=True, trainer=DefaultTrainer) -> Tuple[torch.nn.Module, CfgNode]:
    model = trainer.build_model(cfg)
    if load_weights:
        # We set resume to false because we want to explicitly load the weights in the cfg.MODEL.WEIGHTS
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    return model, cfg


def random_dataset_data(dataset_dict, metadata_dict, rand_samples, seed=1001):
    """
        Get a list of randomly sampled data points from the dataset_dict
        TODO(): This can probably be done by default by some PyTorch / Detectron2 function.

    :param dataset_dict: Detectron2 dataset dict for the dataset
    :param metadata_dict: Detectron2 metadata for the dataset
    :param rand_samples: Number of samples to return.
    :param seed: Seed for generating the random sample. Fixed for determinism. Chose 1001 based on decent images

    TODO(): Mention return vals
    """
    random.seed(seed)
    random_dataset_sample = [f for f in random.sample(dataset_dict, rand_samples)]
    # The ground truths that should be detected for the images, but we remove the gt's which have the 'iscrowd' file.
    gt_detections_per_img = [[metadata_dict.thing_classes[anno['category_id']]
                              for anno in sample['annotations'] if anno['iscrowd'] == 0]
                             for sample in random_dataset_sample
                             ]
    return random_dataset_sample, gt_detections_per_img


def overlay_dataset_imgs(dataset_dict, dataset_metadata, scale) -> List:
    """
    Returns rand_number imgs randomly chosen from the dataset_dict. The images are read by OpenCV
    and can be displayed using the utils in 'dutils' 'imageutils'.
    :return: List of imgs ready to be displayed via cv2 utilities in imageutils
    :param dataset_dict: Detectron2 dataset
    :param dataset_metadata: Detectron2 metadata for the dataset
    :param rand_number: Number of images to display
    """
    imgs = []
    for d in dataset_dict:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=scale)
        vis = visualizer.draw_dataset_dict(d)
        imgs.append(vis.get_image()[:, :])
    return imgs

def overlay_all_dataset_with_color(dataset_dict, dataset_metadata, scale, color, alpha = 0.5 ) -> List:
    """
    Returns rand_number imgs randomly chosen from the dataset_dict. The images are read by OpenCV
    and can be displayed using the utils in 'dutils' 'imageutils'.
    :return: List of imgs ready to be displayed via cv2 utilities in imageutils
    :param dataset_dict: Detectron2 dataset
    :param dataset_metadata: Detectron2 metadata for the dataset
    :param rand_number: Number of images to display
    """
    imgs = []
    assigned_colors = [color for _ in range(len(dataset_metadata.thing_classes))]
    for d in dataset_dict:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=scale)
        masks = [x['segmentation'] for x in d['annotations']]
        vis = visualizer.overlay_instances(masks=masks, assigned_colors = assigned_colors, alpha = alpha)
        imgs.append(vis.get_image()[:, :])
    return imgs

def coco_clsid_to_name(clsid, metadata):
    thing_classes = metadata.thing_classes
    coco_id_to_contiguous_id = metadata.thing_dataset_id_to_contiguous_id
    return thing_classes[coco_id_to_contiguous_id[clsid]]

def detectron_clsid_to_name(clsid, metadata):
    thing_classes = metadata.thing_classes
    return thing_classes[clsid]
