# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os
import copy

from detectron2.data import MetadataCatalog, DatasetCatalog
from .register_coco import register_coco_instances, register_coco_panoptic_separated
from .lvis import register_lvis_instances, get_lvis_instances_meta
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .pascal_voc import register_pascal_voc
from .builtin_meta import _get_builtin_metadata
from .meta_coco import register_meta_coco

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014_100.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}

_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root="datasets"):
    """
        Register both 'classic' COCO datasets and few-shot variants.
    """
    # Register normal COCO datasets
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
    # Register 'meta' datasets (novel, base, all etc)
    # Splits from https://github.com/ucbdrive/few-shot-object-detection
    METASPLITS = [  # name, img_path, json_path
        ("coco_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco_trainval_base", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco_trainval_novel", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
    ]

    for dataset_split in [0, 1]:
        # All metadata for premade base/novel/all categories for the COCO dataset is generated at once, only once
        meta_metadata = _get_builtin_metadata("coco_fewshot", dataset_split)
        # COCO can have multiple 60-20 splits (just two for now)
        split_name = '' if dataset_split == 0 else f'_split{dataset_split}'
        for name, image_root, json_file in METASPLITS:
            # Allow modifying the meta-data in the register function.
            copied_metadata = copy.deepcopy(meta_metadata)

            register_meta_coco(name + split_name,
                               copied_metadata,
                               os.path.join(
                                   root, json_file) if "://" not in json_file else json_file,
                               os.path.join(root, image_root))

        FEW_SHOT_SPLITS = []
        # register small meta datasets for fine-tuning stage
        for prefix in ["all", "novel"]:
            for shot in [1, 2, 3, 5, 10, 30]:
                for seed in range(10):
                    seed_str = "" if seed == 0 else "_seed{}".format(seed)
                    name = "coco_trainval_{}_{}shot{}".format(prefix, shot, seed_str)
                    name += split_name
                    FEW_SHOT_SPLITS.append((name, "coco/trainval2014", ""))

                    # HACK(): Hack added afterwards to allow for seed0/ and unify all scripts.
                    if seed == 0:
                        seed_str = "_seed{}".format(seed)
                        name = "coco_trainval_{}_{}shot{}".format(prefix, shot, seed_str)
                        name += split_name
                        FEW_SHOT_SPLITS.append((name, "coco/trainval2014", ""))

        for name, image_root, json_file in FEW_SHOT_SPLITS:
            # Allow modifying the meta-data in the register function.
            copied_metadata = copy.deepcopy(meta_metadata)

            register_meta_coco(name,
                               copied_metadata,
                               os.path.join(
                                   root, json_file) if "://" not in json_file else json_file,
                               os.path.join(root, image_root))

    # Register VOC Validation set. Only used for testing
    register_coco_instances('VOCSegm_val_novel', dict(
    ), 'datasets/VOCOutput/annotations/val_converted.json', 'datasets/VOCOutput/val/')
    meta_voc = MetadataCatalog.get('VOCSegm_val_novel')

    #TODO: This is hacky but time is short
    meta_voc.novel_dataset_id_to_contiguous_id = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19}
    meta_voc.novel_classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'boat', 'bird',
                              'cat', 'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 'sofa', 'pottedplant', 'diningtable', 'tvmonitor']

    # Register VOC Validation set with few-shot splits.
    # VOC Metadata needed for base, novel, all
    # meta_metadata = _get_builtin_metadata("coco_fewshot", dataset_split)


# ==== Predefined datasets and splits for LVIS ==========

_PREDEFINED_SPLITS_LVIS = {
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/train2017", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/val2017", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/val2017", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/test2017", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/train2017", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/val2017", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root="datasets"):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========


_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val", "cityscapes/gtFine/val"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test", "cityscapes/gtFine/test"),
}


def register_all_cityscapes(root="datasets"):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="sem_seg", **meta
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root="datasets"):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2007_test_2012_eval", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        # TODO: Hack to test 2012 eval mode on 2007 dataset
        # FIXME: Don't leave this like this
        if name == 'voc_2007_test_2012_eval':
            year = 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"



# Register them all under "./datasets"
register_all_coco()
register_all_lvis()
register_all_cityscapes()
register_all_pascal_voc()
