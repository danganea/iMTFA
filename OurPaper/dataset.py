import os
from collections import Counter
from typing import List, Dict
from OurPaper.myconstants import *
import numpy as np
from detectron2.data import MetadataCatalog, DatasetCatalog, \
    build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets.coco import load_coco_json, convert_to_coco_json
from detectron2.data.datasets.meta_coco import load_meta_coco_json


def get_COCO_subsplit(classes):
    """
    Given a list of  category id's present in the original COCO category, return a list of

    Example:
    [{"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"}, ...]
    """
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

    # Get all classes in COCO categories which pertain to the given class indices in 'classes'
    filtered_list = [cls for cls in COCO_CATEGORIES if cls['id'] in classes]
    # Warn that some given classes have been ignored due to them not being in COCO at all
    filtered_classes = [cls['id'] for cls in filtered_list]
    ignored_classes = [cls for cls in classes if cls not in filtered_classes]
    if len(ignored_classes) != 0:
        print(f'Ignored the following classes: {ignored_classes}')

    # TODO(): Perhaps turn this into some assert.
    # Raise an error if any of the classes have the 'isthing' attribute, since they are not supported
    for cls in filtered_list:
        if cls['isthing'] == 0:
            raise ValueError('given classes contain isthing = 0, but isthing not supported')
    return filtered_list


def build_metadata_from_subsplit_COCO(class_indices):
    filtered_COCO_classes = get_COCO_subsplit(class_indices)

    thing_ids = [cls['id'] for cls in filtered_COCO_classes if cls['isthing'] == 1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [cls['name'] for cls in filtered_COCO_classes if cls['isthing'] == 1]
    thing_colors = [cls['color'] for cls in filtered_COCO_classes if cls['isthing'] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "classes_to_eval": class_indices
    }
    return ret


def register_test_class_coco(name, class_indices, json_path='cocosplit/datasplit/trainvalno5k.json',
                             imgroot='coco/trainval2014'):
    # TODO(): Hardcoded datasets root folder
    json_path, imgroot = os.path.join('datasets', json_path), os.path.join('datasets', imgroot)
    meta = build_metadata_from_subsplit_COCO(class_indices)
    print(f"Registering dataset named {name} having the following classes: {meta['thing_classes']}")
    DatasetCatalog.register(name, lambda: load_meta_coco_json(json_path, imgroot, meta, name))

    # Set the meta-data
    MetadataCatalog.get(name).set(
        json_file=json_path, image_root=imgroot, evaluator_type="coco",
        dirname="datasets/coco", **meta,
    )


def _get_novel_base_splits_coco(novel_classes_split=0, novel_cls_idx=None, base_cls_idx=None):
    """
        Code to get the 4 possible COCO-splits according to "One-Shot Instance Segmentation"
    """
    if novel_cls_idx is None:  # Class indices not provided, build them from the split
        if base_cls_idx is not None:
            raise ValueError('base_cls_idx is not None while novel_cls_idx is!')
        if novel_classes_split >= COCO_MAX_NOVEL_SPLIT or novel_classes_split < 0:
            raise ValueError(f'Dataset only supports {COCO_MAX_NOVEL_SPLIT} possible one_shot splits')

        index = novel_classes_split
        step_size = COCO_ALL_CLASSES_NUMBER / COCO_CLASSES_PER_SPLIT

        novel_cls_idx = np.array([step_size * i + index for i in range(COCO_CLASSES_PER_SPLIT)])
        # The base classes are just the other classes of the dataset.
        base_cls_idx = np.array(range(COCO_ALL_CLASSES_NUMBER))[
            np.array([i not in novel_cls_idx for i in range(COCO_ALL_CLASSES_NUMBER)])]

    return novel_cls_idx, base_cls_idx


def prepare_coco_fewshot(coco_dataset, output_dir, novel_classes_split=0, novel_cls_idx=None, base_cls_idx=None):
    """
    Given a coco_dataset supported by Detectron2 ( coco_2017_train, coco_2017_val etc) load it's JSON file and create
    another JSON file that only includes the
    """
    novel_cls_idx, base_cls_idx = _get_novel_base_splits_coco(novel_classes_split, novel_cls_idx, base_cls_idx)

    # img_root, json_file = PREDEFINED_COCO_PATHS[]
    img_root, json_file = os.path.join(OURPAPER_DATASET_FOLDER, img_root), os.path.join(OURPAPER_DATASET_FOLDER, json_file)


def get_dataset_dict_coco(dataset_path=None, novel_classes_split=0, novel_cls_idx=None, base_cls_idx=None):
    """
    TODO(): Description
    Load the COCO 2017 dataset.
    """
    if dataset_path is None:
        raise ValueError('No dataset path provided!')

    if novel_cls_idx is None:  # Class indices not provided, build them from the split
        if base_cls_idx is not None:
            raise ValueError('base_cls_idx is not None while novel_cls_idx is!')
        if novel_classes_split >= COCO_MAX_NOVEL_SPLIT or novel_classes_split < 0:
            raise ValueError(f'Dataset only supports {COCO_MAX_NOVEL_SPLIT} possible one_shot splits')

        index = novel_classes_split
        step_size = COCO_ALL_CLASSES_NUMBER / COCO_CLASSES_PER_SPLIT

        novel_cls_idx = np.array([step_size * i + index for i in range(COCO_CLASSES_PER_SPLIT)])
        # The base classes are just the other classes of the dataset.
        base_cls_idx = np.array(range(COCO_ALL_CLASSES_NUMBER))[
            np.array(i not in novel_classes_split for i in range(COCO_ALL_CLASSES_NUMBER))]


def custom_dataset_dict(json_file, img_root, name, subset='train'):
    # Basically try out register_coco_instances but without the metadata.
    dataset_dicts = load_coco_json(json_file, img_root, name)
    # Logic for filtering dataset_dicts can go here. This includes omitting some classes and such.

    # End of logic
    return dataset_dicts


def custom_register_dataset(name, train=True, metadata={}):
    base_dataset = 'coco_2017'  # Assume test datasets always based on coco_2017 for now
    if train:
        base_dataset = base_dataset + '_train'
        name = name + '_train'
    else:
        base_dataset = base_dataset + '_val'
        name = name + '_val'

    img_root, json_file = PREDEFINED_COCO_PATHS[base_dataset]
    img_root, json_file = os.path.join(OURPAPER_DATASET_FOLDER, img_root), os.path.join(OURPAPER_DATASET_FOLDER, json_file)

    DatasetCatalog.register(name, lambda: custom_dataset_dict(json_file, img_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=img_root, evaluator_type="coco", **metadata
    )


get_dataset_dict = get_dataset_dict_coco
# get_dataset_dict = get_dataset_dict_coco
