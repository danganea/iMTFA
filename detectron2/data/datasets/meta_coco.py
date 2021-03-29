import io
import logging
import contextlib
import os
import datetime
import json
import numpy as np
import re

from PIL import Image

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from fvcore.common.file_io import PathManager, file_lock

from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)


def load_meta_coco_json(json_file, image_root, metadata, dataset_name, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format, removing instances which do not appear in the metadata
    classes.
    Currently supports instance detection and instance segmentation.

    Similar to 'load_coco_json' but has 'metadata' be explicitly passed.
    This allows the direct removal from the loaded ground truths of annotations which are not
    included in the metadata. This removal is done based on the:

    metadata['thing_dataset_id_to_contiguous_id'] field

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        metadata (dict): Metadata
        dataset_name (str): Dataset name, unused.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    def get_dataset_options(regex, dataset_name):
        opt = re.findall(regex, dataset_name)
        assert len(opt) == 1
        opt = int(opt[0])
        return opt


    is_shots = 'shot' in dataset_name
    if is_shots:  # Few shot dataset
        fileids = {}
        split_dir = os.path.join('datasets', 'cocosplit')
        if 'seed' in dataset_name:  # If seed is in the file_name, the shots are all in jsons in subdirs like seed1/
            # seed2/
            shot = get_dataset_options('_(\d+)shot_', dataset_name)
            seed = get_dataset_options('_seed(\d+)', dataset_name)
            split_dir = os.path.join(split_dir, 'seed{}'.format(seed))
        else:  # Else default is in datasets/cocosplit with the files being in that folder.
            shot = dataset_name.split('_')[-1].split('shot')[0]

        # For every class in 'thing_classes', load the few-shot images that we have for it and store it in a dict.
        for idx, cls in enumerate(metadata['thing_classes']):
            json_file = os.path.join(
                split_dir, 'full_box_{}shot_{}_trainval.json'.format(shot, cls))
            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)

            # sort indices for reproducible results
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            fileids[idx] = list(zip(imgs, anns))
    else:  # Normal, not few-shot dataset.
        timer = Timer()
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))

    id_map = metadata['thing_dataset_id_to_contiguous_id']

    if is_shots:
        image_number = sum([1 for k, v in fileids.items() for i, a in v])
        shot_number = sum([len(a) for k, v in fileids.items() for i, a in v])
        logger.info('Loaded {} images and {} shots'.format(image_number, shot_number))
    else:
        logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    if is_shots:
        for _, fileids_ in fileids.items():
            dicts = []
            for (img_dict, anno_dict_list) in fileids_:
                for anno in anno_dict_list:
                    record = {}
                    record['file_name'] = os.path.join(image_root,
                                                       img_dict['file_name'])
                    record['height'] = img_dict['height']
                    record['width'] = img_dict['width']
                    image_id = record['image_id'] = img_dict['id']

                    assert anno['image_id'] == image_id
                    assert anno.get('ignore', 0) == 0

                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    segm = anno.get("segmentation", None)
                    if segm:  # either list[list[float]] or dict(RLE)
                        if not isinstance(segm, dict):
                            # filter out invalid polygons (< 3 points)
                            segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                            if len(segm) == 0:
                                num_instances_without_valid_segmentation += 1
                                continue  # ignore this instance
                        obj["segmentation"] = segm

                    obj['bbox_mode'] = BoxMode.XYWH_ABS
                    obj['category_id'] = id_map[obj['category_id']]
                    record['annotations'] = [obj]
                    dicts.append(record)

            if len(dicts) > int(shot):
                logger.warning("len(dicts) > int(shot) reached, handling it but it shouldn't happen!")
                dicts = np.random.choice(dicts, int(shot), replace=False)
            dataset_dicts.extend(dicts)
    else:
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(image_root, img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                # Check that the image_id in this annotation is the same as
                # the image_id we're looking at.
                # This fails only when the data parsing logic or the annotation file is buggy.

                # The original COCO valminusminival2014 & minival2014 annotation files
                # actually contains bugs that, together with certain ways of using COCO API,
                # can trigger this assertion.
                assert anno["image_id"] == image_id

                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                segm = anno.get("segmentation", None)
                if segm:  # either list[list[float]] or dict(RLE)
                    if not isinstance(segm, dict):
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    obj["segmentation"] = segm

                obj["bbox_mode"] = BoxMode.XYWH_ABS

                # Only include classes in the metadata mapping for the annotations.
                # This allows to only load 'base', 'novel' classes etc.
                if obj['category_id'] in id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def register_meta_coco(name, metadata, annofile, imgdir):
    """
    Register a metadata dataset (i.e. having 'base', 'novel', 'all' as possible substrings in the name)

    Args:
        name: Dataset name to register
        metadata:  Metadata dictionary with 'thing_classes' but also 'base_classes', 'novel_classes' etc.
        imgdir:
        annofile:
    """

    # Change the metadata depending if 'name' refers to 'base' or 'novel'
    if '_base' in name or '_novel' in name:
        split = 'base' if '_base' in name else 'novel'
        metadata['thing_dataset_id_to_contiguous_id'] = \
            metadata['{}_dataset_id_to_contiguous_id'.format(split)]
        metadata['thing_classes'] = metadata['{}_classes'.format(split)]

    # TODO(): Do logging here.
    # print(f'Registering meta-data for {name} with  {len(metadata["thing_classes"])} number of classes ')

    # Register a function to load the dataset. Note: Function is not called here, but lazily called when loading.
    DatasetCatalog.register(
        name, lambda: load_meta_coco_json(annofile, imgdir, metadata, name),
    )

    class_list = list(metadata['thing_dataset_id_to_contiguous_id'].keys())

    #TODO(): Refactor
    if 'coco' in name:
        dirname = 'datasets/coco'
    else:
        raise ValueError
    # Set the meta-data
    MetadataCatalog.get(name).set(
        json_file=annofile, image_root=imgdir, evaluator_type="coco",
        dirname=dirname, **metadata, classes_to_eval=class_list
    )


def register_meta_coco_test():
    raise NotImplementedError()
