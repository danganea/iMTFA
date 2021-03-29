import torch


mdl = torch.load('checkpoints/coco/faster_rcnn/fsdet_novel_random/model_final.pth')



def zero_component(model,component_name):
    model['model'][component_name] = torch.zeros_like(model['model'][component_name])
    return model


mdl = zero_component(mdl,'roi_heads.box_predictor.bbox_pred.bias')
mdl = zero_component(mdl,'roi_heads.box_predictor.bbox_pred.weight')

mdl = zero_component(mdl,'roi_heads.box_predictor.cls_score.bias')
mdl = zero_component(mdl,'roi_heads.box_predictor.cls_score.weight')


mdl = zero_component(mdl,'roi_heads.mask_head.predictor.bias')
mdl = zero_component(mdl,'roi_heads.mask_head.predictor.weight')


torch.save(mdl, 'zeroed_out_all_novel.pth')


print(mdl['model']['roi_heads.mask_head.predictor.bias'])
