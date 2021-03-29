# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn
from enum import Enum

from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

from typing import Tuple

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]

logger = logging.getLogger(__name__)

class _EvalMethod(Enum):
    DEFAULT = 0,
    LIKE_ONESHOT = 1,
    LIKE_FGN = 2

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """


    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT
        self.gt_test_proposals = cfg.MODEL.GT_TEST_PROPOSALS
        self.eval_method = _EvalMethod[cfg.TEST.EVAL_METHOD.upper()]

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        from detectron2.layers import FrozenBatchNorm2d
        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Also freeze batchnorm, not done in original code
            # TODO(): Is this even correct, or does it just return a new module?
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.backbone)
            logger.info('froze backbone parameters')
            print('froze backbone parameters')

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            logger.info('froze proposal generator parameters')
            print('froze proposal generator parameters')
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.proposal_generator)

        if cfg.MODEL.ROI_HEADS.FREEZE_BOX_HEAD:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.roi_heads.box_head)
            logger.info('froze roi_box_head parameters')
            print('froze roi_box_head parameters')

        if cfg.MODEL.ROI_MASK_HEAD.FREEZE:
            for p in self.roi_heads.mask_head.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.roi_heads.mask_head)
            logger.info('froze roi mask head parameters')

        if cfg.MODEL.ROI_MASK_HEAD.FREEZE_WITHOUT_PREDICTOR \
                and cfg.MODEL.ROI_MASK_HEAD.FREEZE:
            # Both frozen doesn't make sense and likely indicates that we forgot to
            # modify a config, so better to early error.
            assert False

        if cfg.MODEL.ROI_MASK_HEAD.FREEZE_WITHOUT_PREDICTOR:
            frozen_names = []
            for n, p in self.roi_heads.mask_head.named_parameters():
                if 'predictor' not in n:
                    p.requires_grad = False
                    frozen_names.append(n)

            logger.info('froze roi mask head parameters without predictor')
            logger.info(f'Names of frozen layers: {frozen_names}')

        self.to(self.device)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def feature_extract_gts_one_image(self, image_dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the feature extraction for the ground_truth proposals in one image.
        Essentially, for one image the following steps are computed:
        - Feature extraction through model backbone.
        - ROI Pooling for each ground truth proposal via `box_pooler`
        - box_head applied to each proposal.

        The result of the box-head is thus the "feature" for each ROI pertaining to each proposal.
        Typically this is a 1024 valued vector

        :param image_dict: Image dict which contains both 'image' and 'instances' fields. Likely produced by `SimpleDatasetMapper`
        :return: Tuple [The feature extractions for each GT ROI, The ground truth classes for them]
        """
        assert 'instances' in image_dict
        assert 'image' in image_dict
        gt_instances = [image_dict['instances'].to(self.device)]

        images = self.preprocess_image([image_dict])
        features = self.backbone(images.tensor)

        feats = [features[f] for f in self.roi_heads.in_features]

        box_features = self.roi_heads.box_pooler(feats, [x.gt_boxes for x in gt_instances])
        box_features = self.roi_heads.box_head(box_features)

        return box_features, image_dict['instances'].gt_classes

    
    def feature_extract_gts_multiple_images(self, batched_inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the feature extraction for the ground_truth proposals in one image.
        Essentially, for one image the following steps are computed:
        - Feature extraction through model backbone.
        - ROI Pooling for each ground truth proposal via `box_pooler`
        - box_head applied to each proposal.

        The result of the box-head is thus the "feature" for each ROI pertaining to each proposal.
        Typically this is a 1024 valued vector

        :param image_dict: Image dict which contains both 'image' and 'instances' fields. Likely produced by `SimpleDatasetMapper`
        :return: Tuple [The feature extractions for each GT ROI, The ground truth classes for them]
        """
        assert 'instances' in batched_inputs[0]
        assert 'image' in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        feats = [features[f] for f in self.roi_heads.in_features]

        box_features = self.roi_heads.box_pooler(feats, [x.gt_boxes for x in gt_instances])
        box_features = self.roi_heads.box_head(box_features)

        return box_features

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator and not self.gt_test_proposals:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            gt_classes = None
            if not self.training and (self.eval_method != _EvalMethod.DEFAULT): 
                assert len(batched_inputs) == 1 #Ensure we're only evaluating on one image
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs][0]
                gt_classes = gt_instances.gt_classes.unique()

            results, _ = self.roi_heads(images, features, proposals, None, gt_classes)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
