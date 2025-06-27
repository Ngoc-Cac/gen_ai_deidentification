import numpy as np, torch
import yaml

from diffusers import AutoPipelineForInpainting

from app_utils import resize_image, dilate_mask
from lama_inpaint import build_lama_model, inpaint_img_with_builded_lama
from tesseract import get_masked_ocr, get_bboxes
from segment_anything.segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
    SamPredictor,
)


from typing import Literal



_ckpt_paths = yaml.safe_load(open('./app/config.yml'))
_model = {
    'lama': None,
    'sam': None,
    'sd': None,
    'inpaint_type': None,
    'device': None,
}

def load_ckpt(
    model_type: Literal['lama', 'sd', 'sam'],
    ckpt_path: str | None = None,
    config_path: str | None = None,
    device: Literal['cuda', 'cpu'] = 'cpu'
):
    if config_path is None:
        config_path = _ckpt_paths['lama_config']
    if ckpt_path is None:
        ckpt_path = _ckpt_paths[model_type]

    if model_type == 'lama':
        _model['sd'] = None
        _model['inpaint_type'] = model_type
        _model['device'] = device
        _ckpt_paths['lama_config'] = config_path
        _model['lama'] = build_lama_model(
            config_path, ckpt_path,
            device=device, weights_only=False
        )
    elif model_type == 'sd':
        _model['lama'] = None
        _model['inpaint_type'] = model_type
        _model['device'] = device
        _model['sd'] = AutoPipelineForInpainting.from_pretrained(
            ckpt_path, safety_checker=None, requires_safety_checker=False
        ).to(device)
    elif model_type == 'sam':
        sam_reg = sam_model_registry['vit_h'](checkpoint=ckpt_path).to(device=device)
        _model['sam'] = {
            'predictor': SamPredictor(sam_reg),
            'generator': SamAutomaticMaskGenerator(sam_reg)
        }


def get_sam_feat(img):
    sam_alias = _model['sam']['predictor']

    sam_alias.set_image(img)
    features = sam_alias.features
    orig_h = sam_alias.orig_h
    orig_w = sam_alias.orig_w
    input_h = sam_alias.input_h
    input_w = sam_alias.input_w
    sam_alias.reset_image()
    return features, orig_h, orig_w, input_h, input_w

def get_click_mask(
    clicked_points, features,
    orig_h, orig_w, input_h, input_w,
    dilate_kernel_size=None,
):
    sam_alias = _model['sam']['predictor']
    sam_alias.is_image_set = True
    sam_alias.features = features
    sam_alias.orig_h = orig_h
    sam_alias.orig_w = orig_w
    sam_alias.input_h = input_h
    sam_alias.input_w = input_w
    
    # Separate the points and labels
    points, labels = zip(*[(point[:2], point[2]) for point in clicked_points])

    # Convert the points and labels to numpy arrays
    input_point = np.array(points)
    input_label = np.array(labels)

    masks, _, _ = sam_alias.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
    else:
        masks = [mask for mask in masks]

    return masks

def get_all_mask(np_image):
    return _model['sam']['generator'].generate(np_image)


def inpaint_image(
    image, mask,
    resolution: int = 512,
    sd_inference_step: int = 50,
    # model_type: Literal['lama', 'sd'] = _model['inpaint_type']
):
    model_type = _model['inpaint_type']
    if model_type == 'lama':
        img_inpainted = inpaint_img_with_builded_lama(
            _model['lama'], image, mask,
            _ckpt_paths['lama_config'], device=_model['device']
        )
    elif model_type == 'sd':
        resized_img = resize_image(image, resolution) / 255
        resized_mask = resize_image(mask, resolution)
        img_inpainted = _model['sd'](
            "remove", output_type='pil',
            image=resized_img, mask_image=resized_mask,
            num_inference_steps=sd_inference_step
        ).images[0].resize((image.shape[1], image.shape[0]))

    return img_inpainted

def clean_phi(
    image,
    detailed_bbox: bool = False,
    resegment: bool = False,
):
    if resegment:
        sam_alias = _model['sam']['predictor']
        sam_alias.set_image(image)

        input_boxes = torch.tensor(
            get_bboxes(image, detailed_bbox),
            device=sam_alias.device
        )
        transformed_boxes = sam_alias.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, mask_input = sam_alias.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        mask = masks.cpu().any(dim=0).squeeze().numpy().astype(np.uint8) * 255
    else:
        mask = get_masked_ocr(image, detailed_bbox)

    if _model['inpaint_type'] == 'sd':
        load_ckpt('lama')
        inpainted_image = inpaint_image(image, mask)
        load_ckpt('sd')
    else:
        inpainted_image = inpaint_image(image, mask)

    return inpainted_image, mask