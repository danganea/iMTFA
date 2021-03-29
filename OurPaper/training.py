import os

import cv2
from collections import defaultdict
import logging
import copy

import torch, torchvision
import detectron2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.dataset_mapper import SimpleDatasetMapper

import dutils.detectronutils as ddu
from OurPaper import BBOX_PRED_COORD_NUMBER
import functools

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=128)
def _get_fewshot_model_weights(model_src: str):
    """
    :returns cls_weights, bbox_pred_weights, bbox_pred_bias
    :rtype: tuple
    """
    mdl = torch.load(model_src)['model']
    few_shot_cls_weights = mdl['roi_heads.box_predictor.cls_score.weight']
    few_shot_bbox_pred_weights = mdl['roi_heads.box_predictor.bbox_pred.weight']
    few_shot_bbox_pred_bias = mdl['roi_heads.box_predictor.bbox_pred.bias']

    return few_shot_cls_weights, few_shot_bbox_pred_weights, few_shot_bbox_pred_bias


def _compute_averaged_shot_representatives(few_shot_dataset, feature_extractor, cfg):
    few_shot_dict = DatasetCatalog.get(few_shot_dataset)
    # Build a dict where each entry contains a list of all the shots for one class
    class_to_img = defaultdict(lambda: [])
    for shot in few_shot_dict:
        class_to_img[shot['annotations'][0]['category_id']].append(shot)
    class_number = len(class_to_img)

    shots_per_img = len(class_to_img[0])
    # Check that we have the same amount of shots for every class
    for v in class_to_img.values():
        assert (len(v) == shots_per_img)
    # Compute class representatives
    simple_mapper = SimpleDatasetMapper(cfg)
    class_representatives = []
    with torch.no_grad():  # Potentially not needed if we set model to eval mode
        for idx in range(class_number):  # Loop over all the classes
            examples_per_shot = 1
            if cfg.INPUT.USE_TRANSFORM_METRIC_AVERAGING:
                examples_per_shot = cfg.INPUT.EXAMPLES_PER_SHOT
            all_feature_list = []
            for _ in range(examples_per_shot):
                feature_list = [feature_extractor.feature_extract_gts_one_image(simple_mapper(shot)) for shot in
                                class_to_img[idx]]
                feature_list = [feature_repr for feature_repr, _ in feature_list]  # only use feature representative
                if cfg.INPUT.NORMALIZE_SHOTS:
                    feature_list = [feature_repr / torch.norm(feature_repr) for feature_repr in feature_list]  # only use feature representative
                all_feature_list.extend(feature_list)
            class_representative = sum(all_feature_list) / len(all_feature_list)
            class_representatives.append(class_representative)
    return class_representatives


def _write_model_results(trained_model: dict, cfg):
    logger.info(f'Writing last_checkpoint file to {cfg.OUTPUT_DIR}')
    with open(os.path.join(cfg.OUTPUT_DIR, 'last_checkpoint'), 'w') as last_checkpoint_file:
        last_checkpoint_file.write('model_final.pth')

    with open(os.path.join(cfg.OUTPUT_DIR, 'model_final.pth'), 'wb') as model_file:
        logger.info(f'Writing model_final.pth')
        torch.save(trained_model, model_file)
        if 'all' in cfg.DATASETS.TEST[0]:
            logger.info(f'Using *all* dataset, writing model_reset_combine.pth as well.\n'
                        f'This is identical to model_final.pth, just renamed.')
            # torch.save(trained_model, 'model_reset_combine.pth')  # Naming needed for run_experiments.py file


#TODO(): Cache mdl?
# TODO(): Horrible hacky API cause we later on added the ability to get the mask weights but
def get_lastlayer_mask_weights(mdl):
    if isinstance(mdl, str):
        mdl = torch.load(mdl)['model']
    elif isinstance(mdl, dict):
        if 'model' in mdl:
            mdl = mdl['model']
    else:
        raise TypeError('Expected dict or str')

    mask_head_weights = mdl['roi_heads.mask_head.predictor.weight']
    mask_head_bias =  mdl['roi_heads.mask_head.predictor.bias']

    return  mask_head_weights, mask_head_bias



def get_lastlayer_cls_score_bias(mdl):
    if isinstance(mdl, str):
        mdl = torch.load(mdl)['model']
    elif isinstance(mdl, dict):
        if 'model' in mdl:
            mdl = mdl['model']
    else:
        raise TypeError('Expected dict or str')

    cls_weights_bias = mdl['roi_heads.box_predictor.cls_score.bias']

    return cls_weights_bias

# we didn't want to break existing code
def get_lastlayers_model_weights(mdl, is_mask_class_specific = False, is_fc_lastlayer = False):
    """
    Returns the weights at the head of the ROI predictor -> classification and bounding box regressor weights + bias
    :returns cls_weights, bbox_pred_weights, bbox_pred_bias
    :rtype: tuple
    """
    if isinstance(mdl, str):
        mdl = torch.load(mdl)['model']
    elif isinstance(mdl, dict):
        if 'model' in mdl:
            mdl = mdl['model']
    else:
        raise TypeError('Expected dict or str')
    cls_weights = mdl['roi_heads.box_predictor.cls_score.weight']
    bbox_pred_weights = mdl['roi_heads.box_predictor.bbox_pred.weight']
    bbox_pred_bias = mdl['roi_heads.box_predictor.bbox_pred.bias']

    return cls_weights, bbox_pred_weights, bbox_pred_bias


# TODO(): Unify with metric-learning code. A lot of code here is the same as there.
# TODO(): Currently unused.
def combine_base_novel_models(base_model, novel_model, metadata_dict):
    """
        Combine base_model and novel_model's final classification weights to obtain a new model.

        Reimplementation of similar code in metric_learning function.

        Does not handle class representatives or 'novel' mode.
    :param base_model:
    :param novel_model:
    :param metadata_dict:
    :return:
    """
    if isinstance(base_model, str):
        base_model = torch.load(base_model)

    if isinstance(novel_model, str):
        novel_model = torch.load(novel_model)

    all_classes = []
    all_classes += list(metadata_dict.base_dataset_id_to_contiguous_id.keys())
    all_classes += list(metadata_dict.novel_dataset_id_to_contiguous_id.keys())
    all_classes = sorted(all_classes)
    final_class_number = len(all_classes)

    # ID_MAP tells you the position in the new trained model of every class.
    # ID_MAP[1] -> 0 will mean that the class with index 1 will go to weight position 0.
    ID_MAP = {v: k for k, v in enumerate(all_classes)}
    # Map previous position in base model to class_id
    BASE_ID_MAP = {v: k for k, v in metadata_dict.base_dataset_id_to_contiguous_id.items()}
    # Map previous position in novel model to class_id
    NOVEL_ID_MAP = {v: k for k, v in metadata_dict.novel_dataset_id_to_contiguous_id.items()}

    base_class_weights, base_bbox_weights, base_bbox_bias = get_lastlayers_model_weights(base_model)
    feature_size = base_class_weights.data.shape[1]

    is_class_agnostic = base_bbox_weights.shape[0] == BBOX_PRED_COORD_NUMBER

    # Initialize new weights
    new_cls_weights = torch.zeros(final_class_number + 1, feature_size)  # Add one for background class
    new_bbox_pred_weights = torch.zeros(final_class_number * 4, feature_size)
    new_bbox_pred_bias = torch.zeros(final_class_number * 4)

    # Fit the base class weights, which we have from training previously, into new training model.
    for idx, class_id in BASE_ID_MAP.items():
        new_weight_pos = ID_MAP[class_id]
        new_cls_weights[new_weight_pos] = base_class_weights.data[idx]

        if not is_class_agnostic:
            new_bbox_pred_weights[new_weight_pos * 4: (new_weight_pos + 1) * 4] = base_bbox_weights.data[
                                                                                  idx * 4: (idx + 1) * 4]
            new_bbox_pred_bias[new_weight_pos] = base_bbox_bias.data[idx]

    # Fit the class representatives into their positions in the new model.
    novel_class_weights, novel_bbox_weights, novel_bbox_bias = get_lastlayers_model_weights(novel_model)
    class_representatives = novel_class_weights[:-1]
    for idx, class_id in NOVEL_ID_MAP.items():
        new_weight_pos = ID_MAP[class_id]
        new_cls_weights[new_weight_pos] = class_representatives[idx]

        if not is_class_agnostic:
            new_bbox_pred_weights[new_weight_pos * 4: (new_weight_pos + 1) * 4] = novel_bbox_weights[
                                                                                  idx * 4: (idx + 1) * 4]
            new_bbox_pred_bias[new_weight_pos] = novel_bbox_bias[idx]

    if is_class_agnostic:
        new_bbox_pred_weights.data = base_bbox_weights.data
        new_bbox_pred_bias = base_bbox_bias.data

    # Last weight of base classes model always goes at the end of our model.
    new_cls_weights[-1] = base_class_weights.data[-1]

    trained_model = {'model': base_model, 'iteration': 0}
    trained_model['model']['roi_heads.box_predictor.cls_score.weight'] = new_cls_weights
    trained_model['model']['roi_heads.box_predictor.bbox_pred.weight'] = new_bbox_pred_weights
    trained_model['model']['roi_heads.box_predictor.bbox_pred.bias'] = new_bbox_pred_bias

    return trained_model


def metric_training(cfg, args) -> dict:
    """
        Use the 'shots' in a few-shot dataset in order to compute feature representations by using the box-head.
        Then, save a model which can perform inference on `CFG.DATASETS.TEST` `as if` the model was trained.
        This means we have a 'model_final' file along with a 'last_checkpoint' file.

        ..note: No trainable parameters here.

        - We use cfg.DATASETS.TRAIN as a dataset containing the FEW-SHOTS.

        - We use cfg.DATASETS.TEST to test on this dataset. cfg.DATASETS.TEST can have a varying amount of classes,
        either for novel or for novel + base

        - We use cfg.MODEL.WEIGHTS as an initial model which has a cosine-similarity based box_head.
        This must therefore be a model which is trained on the 'base' classes.

        - We use args.src1 as the source of a pre-trained few-shot model. Since cfg.MODEL.WEIGHTS keeps a path to a
        model trained on base classes, we need src1 to hold a path to a model trained on the novel classes. Note that
        in the case of average-finetuning this is not used. This setting is also used in case we have a non class-agnostic
        box head.

        Using cfg.DATASETS.TRAIN for the novel classes along with cfg.MODEL.WEIGHTS  we bring the number of
        classes to the amount in cfg.MODEL.TEST, by either only using the novel classes or by adding the base class
        representatives to the novel class representatives
    """
    # This is a training function so don't call it during eval mode
    assert not args.eval_only
    few_shot_dataset = cfg.DATASETS.TRAIN[0]
    assert 'shot' in few_shot_dataset

    metadata_dict = MetadataCatalog.get(few_shot_dataset)
    # Decide based on the test dataset how many classes we want to support.
    novel_class_number = len(metadata_dict.novel_classes)
    base_class_number = len(metadata_dict.base_classes)
    if 'novel' in cfg.DATASETS.TEST[0]:
        final_class_number = novel_class_number
        used_datasets = ['novel']
    elif 'all' in cfg.DATASETS.TEST[0]:
        final_class_number = novel_class_number + base_class_number
        used_datasets = ['base', 'novel']
    else:
        used_datasets = ['base']
        raise ValueError("Only novel and 'all' testing datasets for pure-metric learning supported")

    assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == final_class_number

    base_cfg = copy.deepcopy(cfg)
    base_cfg.defrost()
    base_cfg.MODEL.ROI_HEADS.NUM_CLASSES = base_class_number
    base_model, base_cfg = ddu.build_model_from_cfg(base_cfg)

    # If we've loaded the correct model then it should have the 'base' amount of classes in the box_predictor
    assert base_model.roi_heads.box_predictor.cls_score.weight.data.shape[0] == base_class_number + 1

    class_representatives = []
    if args.method == 'pure-metric-averaged':  # The average of every set of shots per class is the class representative
        class_representatives = _compute_averaged_shot_representatives(few_shot_dataset, base_model, base_cfg)
    elif args.method == 'pure-metric-finetuned' and 'all' in cfg.DATASETS.TEST[0]:  # 'Train' not needed for 'novel'
        # Consider the class representatives to be just the weights in the finetuned model.
        # Only useful when considering the 'all' dataset, to see how good the finetuned-weights transfer is.
        few_shot_cls_weights, _, _ = get_lastlayers_model_weights(args.src1)
        class_representatives = [row for row in few_shot_cls_weights]
        class_representatives = class_representatives[:-1]  # Discard last element as it's the background.
    else:
        assert False

    assert (len(class_representatives) == novel_class_number)

    all_classes = []
    if 'base' in used_datasets:
        all_classes += list(metadata_dict.base_dataset_id_to_contiguous_id.keys())
    if 'novel' in used_datasets:
        all_classes += list(metadata_dict.novel_dataset_id_to_contiguous_id.keys())
    all_classes = sorted(all_classes)

    # ID_MAP tells you the position in the new trained model of every class.
    # ID_MAP[1] -> 0 will mean that the class with index 1 will go to weight position 0.
    ID_MAP = {v: k for k, v in enumerate(all_classes)}
    # Map previous position in base model to class_id
    BASE_ID_MAP = {v: k for k, v in metadata_dict.base_dataset_id_to_contiguous_id.items()}
    # Map previous position in novel model to class_id
    NOVEL_ID_MAP = {v: k for k, v in metadata_dict.novel_dataset_id_to_contiguous_id.items()}

    base_head = base_model.roi_heads.box_predictor
    feature_size = base_head.cls_score.weight.data.shape[1]

    # Initialize new weights
    new_cls_weights = torch.zeros(final_class_number + 1, feature_size)  # Add one for background class

    # We only have a bias if we're using FC
    if cfg.MODEL.ROI_HEADS.OUTPUT_LAYER != "CosineSimOutputLayers":
        new_cls_weights_bias = torch.zeros(final_class_number + 1)  # Add one for background class

    new_bbox_pred_weights = torch.zeros(final_class_number * 4, feature_size)
    new_bbox_pred_bias = torch.zeros(final_class_number * 4)

    # We transfer weights for each mask predictor branch if we don't use a class agnostic mask
    if not cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
        mask_feature_size = base_model.roi_heads.mask_head.predictor.weight.shape[1:]

        new_mask_pred_weights = torch.zeros((final_class_number,*mask_feature_size))
        new_mask_pred_biases = torch.zeros(final_class_number)


    # Fit the base class weights, which we have from training previously, into new training model.
    if 'base' in used_datasets:
        for idx, class_id in BASE_ID_MAP.items():
            new_weight_pos = ID_MAP[class_id]
            new_cls_weights[new_weight_pos] = base_head.cls_score.weight.data[idx]

            if not cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG:
                new_bbox_pred_weights[new_weight_pos * 4: (new_weight_pos + 1) * 4] = base_head.bbox_pred.weight.data[
                                                                                      idx * 4: (idx + 1) * 4]
                new_bbox_pred_bias[new_weight_pos] = base_head.bbox_pred.bias.data[idx]

            if not cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
                new_mask_pred_weights[new_weight_pos] = base_model.roi_heads.mask_head.predictor.weight.data[idx]
                new_mask_pred_biases[new_weight_pos] = base_model.roi_heads.mask_head.predictor.bias.data[idx]

            if cfg.MODEL.ROI_HEADS.OUTPUT_LAYER != "CosineSimOutputLayers":
                new_cls_weights_bias[new_weight_pos] = base_head.cls_score.bias.data[idx]

           
    # Fit the class representatives into their positions in the new model.
    if 'novel' in used_datasets:
        for idx, class_id in NOVEL_ID_MAP.items():
            new_weight_pos = ID_MAP[class_id]
            new_cls_weights[new_weight_pos] = class_representatives[idx]

            if not cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG:
                # If we have a class-specific box-head then we will place novel weights from an
                # existing novel dataset (taken from args.src1) into the novel box_head slots.
                _, few_shot_bbox_pred_weights, few_shot_bbox_pred_bias = get_lastlayers_model_weights(args.src1)
                new_bbox_pred_weights[new_weight_pos * 4: (new_weight_pos + 1) * 4] = few_shot_bbox_pred_weights[
                                                                                      idx * 4: (idx + 1) * 4]
                new_bbox_pred_bias[new_weight_pos] = few_shot_bbox_pred_bias[idx]
            
            if not cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
                fewshot_mask_weights, fewshot_mask_bias = get_lastlayer_mask_weights(args.src1)
                new_mask_pred_weights[new_weight_pos] = fewshot_mask_weights[idx]
                new_mask_pred_biases[new_weight_pos] = fewshot_mask_bias[idx]

            if cfg.MODEL.ROI_HEADS.OUTPUT_LAYER != "CosineSimOutputLayers":
                fewshot_cls_bias = get_lastlayer_cls_score_bias(args.src1)
                new_cls_weights_bias[new_weight_pos] = fewshot_cls_bias[idx]

    if cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG:
        new_bbox_pred_weights.data = base_head.bbox_pred.weight.data
        new_bbox_pred_bias.data = base_head.bbox_pred.bias.data

    # Last weight of base classes model always goes at the end of our model.
    new_cls_weights[-1] = base_head.cls_score.weight.data[-1]

    trained_model = {'model': base_model.state_dict(), 'iteration': 0}
    trained_model['model']['roi_heads.box_predictor.cls_score.weight'] = new_cls_weights
    trained_model['model']['roi_heads.box_predictor.bbox_pred.weight'] = new_bbox_pred_weights
    trained_model['model']['roi_heads.box_predictor.bbox_pred.bias'] = new_bbox_pred_bias

    #TODO(): Don't forget to add  base bg bias and others?
    #Add cls score bias if needed and mask predictor if mask is not cls agnostic
    if not cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
        trained_model['model']['roi_heads.mask_head.predictor.weight'] = new_mask_pred_weights 
        trained_model['model']['roi_heads.mask_head.predictor.bias'] = new_mask_pred_biases 

    if cfg.MODEL.ROI_HEADS.OUTPUT_LAYER != "CosineSimOutputLayers":
        new_cls_weights_bias[-1] = base_head.cls_score.bias.data[-1]
        trained_model['model']['roi_heads.box_predictor.cls_score.bias'] = new_cls_weights_bias

    _write_model_results(trained_model, cfg)

    return trained_model
