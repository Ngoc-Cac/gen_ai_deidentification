import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
os.chdir("../")

import cv2, gradio as gr, numpy as np, torch

from app_utils import HWC3, resize_image, resize_points
from model_backend import (
    clean_phi,
    get_all_mask,
    get_click_mask,
    get_sam_feat,
    inpaint_image,
    load_ckpt,
)


def load_model(model_val, progress=gr.Progress()):
    progress(0.5, 'Loading weights...')

    load_ckpt(model_val, device=device)
    model["inpaint_type"] = model_val
    
    progress(1, 'Loading finished...')
    return model_val, gr.Row(visible=model['inpaint_type'] == 'sd')

def get_selected_mask(all_mask, clicked_points):
    chosen_mask = [
        min([mask for mask in all_mask if mask['segmentation'][y, x]],
            key=lambda ann: ann['area'])['segmentation']
        for x, y in clicked_points
    ]

    mask_img = np.logical_or.reduce(np.stack(chosen_mask, axis=0), axis=0)
    return mask_img.astype(np.uint8) * 255


def segment_by_click(
    x, y,
    original_image, point_prompt, clicked_points,
    image_resolution, dilate_kernel_size,
    features, orig_h, orig_w, input_h, input_w
):
    label = point_prompt
    lab = 1 if label == "Foreground Point" else 0
    clicked_points.append((x, y, lab))

    input_image = np.array(original_image, dtype=np.uint8)
    H, W, C = input_image.shape
    input_image = HWC3(input_image)
    # img = resize_image(input_image, image_resolution)

    # Update the clicked_points
    resized_points = resize_points(
        clicked_points, input_image.shape, image_resolution
    )
    mask_click_np = get_click_mask(
        resized_points, features,
        orig_h, orig_w, input_h, input_w,
        dilate_kernel_size,
    )

    # Convert mask_click_np to HWC format
    mask_click_np = np.transpose(mask_click_np, (1, 2, 0)) * 255.0

    mask_image = HWC3(mask_click_np.astype(np.uint8))
    mask_image = cv2.resize(
        mask_image, (W, H), interpolation=cv2.INTER_LINEAR)

    # Draw circles for all clicked points
    edited_image = input_image
    point_radius = int(min(edited_image.shape[:2]) * 20 / 1741)
    for x, y, lab in clicked_points:
        # Set the circle color based on the label
        color = (255, 0, 0) if lab == 1 else (0, 0, 255)
        # Draw the circle
        edited_image = cv2.circle(edited_image, (x, y), point_radius, color, -1)

    # Set the opacity for the mask_image and edited_image
    opacity_mask = 0.75
    opacity_edited = 1.0

    # Combine the edited_image and the mask_image using cv2.addWeighted()
    overlay_image = cv2.addWeighted(
        edited_image,
        opacity_edited,
        (mask_image *
            np.array([0 / 255, 255 / 255, 0 / 255])).astype(np.uint8),
        opacity_mask,
        0,
    )

    return (
        overlay_image,
        clicked_points,
        mask_image
    )

def select_by_click(
    x, y, all_mask,
    clicked_points, source_image
):
    clicked_points.append((x, y))
    mask_img = get_selected_mask(all_mask, clicked_points)

    point_radius = int(min(source_image.shape[:2]) * 20 / 1741)
    for x, y in clicked_points:
        source_image = cv2.circle(
            source_image, (x, y), point_radius,
            (0, 0, 255), -1
        )
    return source_image, clicked_points, mask_img

def undo_mask_selection(
    seg_all_res, clicked_points
):
    combined_mask = np.array(seg_all_res['overlay'], dtype=np.uint8)
    if len(clicked_points) < 2:
        return combined_mask, [], None
    
    clicked_points.pop()
    mask_img = get_selected_mask(seg_all_res['all_mask'], clicked_points)

    overlay = combined_mask
    point_radius = int(min(overlay.shape[:2]) * 20 / 1741)
    for x, y in clicked_points:
        overlay = cv2.circle(
            overlay, (x, y), point_radius,
            (0, 0, 255), -1
        )
    return overlay, clicked_points, mask_img


def process_image_click(
    source_image, seg_all_res, original_image,
    point_prompt, clicked_points, image_resolution,
    dilate_kernel_size, features,
    orig_h, orig_w, input_h, input_w,
    evt: gr.SelectData
):
    x, y = evt.index
    if seg_all_res is None:
        return segment_by_click(
            x, y, original_image,
            point_prompt, clicked_points,
            image_resolution,
            dilate_kernel_size, features,
            orig_h, orig_w, input_h, input_w,
        )
    else:
        return select_by_click(
            x, y, seg_all_res['all_mask'],
            clicked_points, source_image
        )
    
    # return (
    #     overlay_img,
    #     clicked_points,
    #     mask_img
    # )

def process_seg_all(
    original_image,
    progress=gr.Progress()
):
    np_image = np.array(original_image, dtype=np.uint8)
    progress(.5, 'Getting all masks...')
    # sort the mask by area to avoid overlapping mask with smaller area
    masks = sorted(get_all_mask(np_image),
        key=lambda mask: mask['area'], reverse=True
    )

    combined_mask = np.ones(
        (masks[0]['segmentation'].shape[0],
         masks[0]['segmentation'].shape[1],
         3),
         dtype=np.uint8
    )
    progress(.75, 'Processing masks...')
    for mask in masks:
        color_mask = np.random.randint(256, size=3, dtype=np.uint8)
        combined_mask[mask['segmentation']] = color_mask
    overlay_img = cv2.addWeighted(
        np_image, 1,
        combined_mask, 0.35, 0
    )
    return overlay_img, {'overlay': overlay_img, 'all_mask': masks}, []

def image_upload(image, image_resolution, progress=gr.Progress()):
    if image is not None:
        progress(0, 'Image uploaded...')
        np_image = np.array(image, dtype=np.uint8)

        progress(1 / 3, 'Processing image...')
        np_image = HWC3(np_image)
        np_image = resize_image(np_image, image_resolution)

        progress(2 / 3, 'Getting SaM features...')
        features, orig_h, orig_w, input_h, input_w = get_sam_feat(np_image)

        progress(1, 'Finished...')
        return [], image, None, features, orig_h, orig_w, input_h, input_w
    else:
        return [], None, None, None, None, None, None, None

def get_inpainted_img(
    image, mask, resolution,
    sd_inference_step,
    progress=gr.Progress(True)
):
    if len(mask.shape)==3: mask = mask[:,:,0]

    progress(0, 'Inpainting...')
    inpaint_res = inpaint_image(
        image, mask,
        resolution, sd_inference_step,
        # model['inpaint_type']
    )
    progress(1, 'Finished...')
    return inpaint_res

def process_clean_phi(
    image, detailed_bbox,
    resegement
):
    inpainted_img, mask = clean_phi(image, detailed_bbox, resegement)
    
    mask_img = (
        HWC3(mask) * np.array([0 / 255, 255 / 255, 0 / 255])
    ).astype(np.uint8)
    overlay_img = cv2.addWeighted(
        image, 1,
        mask_img, 0.35, 0
    )
    return (
        inpainted_img,
        mask,
        overlay_img
    )


# build models
device = "cuda" if torch.cuda.is_available() else "cpu"
model = {'inpaint_type': 'lama'}

# build the sam model
load_ckpt('sam', device=device)

# build the lama model
load_ckpt(model['inpaint_type'], device=device)


button_size = (100, 50)
with gr.Blocks() as demo:
    clicked_points = gr.State([])
    origin_image = gr.State(None)
    click_mask = gr.State(None)
    seg_all_res = gr.State(None)
    features = gr.State(None)
    orig_h = gr.State(None)
    orig_w = gr.State(None)
    input_h = gr.State(None)
    input_w = gr.State(None)
    txt_resegement = gr.State(False)
    txt_detailed = gr.State(False)

    with gr.Row():
        with gr.Column(variant='panel'):
            with gr.Tab('Segmentation Panel'):
                with gr.Row(equal_height=True, variant='panel'):
                    with gr.Column(min_width=50):
                        remove_phi = gr.Button(
                            "Remove PHI Text with LaMa", variant='primary')
                        save_clean_phi = gr.Button(
                            'Save Results', variant='secondary')
                    clean_phi_options = gr.CheckboxGroup(
                        [
                            ('Give detailed bounding box', 'detailed'),
                            ('Re-segment with SaM', 'resegment')
                        ], show_label=False, container=False,
                        min_width=50
                    )
                segment_everything = gr.Button("Segment Everything", variant='primary')
                clear_button_image = gr.Button(value="Reset", variant="secondary")
                undo_select = gr.Button("Undo Mask Selection", visible=False)
            with gr.Tab('Inpaint Panel'):
                inpaint_button = gr.Button("Inpaint Image", variant="primary")
                with gr.Row(): gr.Markdown('### Inpainting Model')
                model_selection = gr.Dropdown(
                    [('LaMa', 'lama'), ('Stable Diffusion v2', 'sd')],
                    value=model['inpaint_type'],
                    show_label=False,
                    interactive=True
                )
                with gr.Row(visible=model['inpaint_type'] == 'sd') as sd_config:
                    sd_inference_step = gr.Slider(
                        minimum=10, maximum=100,
                        value=50, step=10
                    )
        with gr.Column(variant='panel', scale=3):
            with gr.Tabs() as image_tab:
                with gr.TabItem('Input Image', id='input'):
                    source_image_click = gr.Image(
                        type="numpy", sources=['upload', 'clipboard'],
                        interactive=True, show_label=False,
                        height=550,
                    )
                    with gr.Row():
                        point_prompt = gr.Radio(
                            choices=["Foreground Point", "Background Point"],
                            value="Foreground Point",
                            label="Point Label",
                            interactive=True,
                            show_label=False,
                        )
                        image_resolution = gr.Slider(
                            label="Image Resolution",
                            minimum=256,
                            maximum=768,
                            value=512,
                            step=64,
                        )
                        dilate_kernel_size = gr.Slider(label="Dilate Kernel Size",
                            minimum=0, maximum=30, step=1, value=3,
                            interactive=True
                        )
                with gr.TabItem('Mask', id='mask'):
                    click_mask = gr.Image(
                        type="numpy", format="png",
                        height=source_image_click.height,
                        interactive=False, show_download_button=True,
                        show_label=False
                    )
                with gr.TabItem('Image Removed with Mask', id='removed'):
                    img_rm_with_mask = gr.Image(
                        type="numpy", format='png',
                        height=source_image_click.height,
                        interactive=False, show_download_button=True,
                        show_label=False,
                    )


    model_selection.change(
        load_model,
        [model_selection],
        [model_selection, sd_config],
    )
    clean_phi_options.change(
        lambda options: ('detailed' in options, 'resegment' in options),
        [clean_phi_options],
        [txt_detailed, txt_resegement],
    )


    source_image_click.upload(
        image_upload,
        inputs=[source_image_click, image_resolution],
        outputs=[clicked_points, origin_image, seg_all_res,
                 features, orig_h, orig_w, input_h, input_w],
        show_progress=True, queue=True
    )
    source_image_click.select(
        process_image_click,
        inputs=[source_image_click, seg_all_res, origin_image,
                point_prompt, clicked_points, image_resolution,
                dilate_kernel_size, features,
                orig_h, orig_w, input_h, input_w],
        outputs=[source_image_click, clicked_points, click_mask],
        show_progress=True,
        queue=True,
    )

    segment_everything.click(
        process_seg_all,
        [origin_image],
        [source_image_click, seg_all_res, clicked_points]
    )
    seg_all_res.change(
        lambda x: gr.Button("Undo Mask Selection", visible=(not x is None)),
        [seg_all_res], [undo_select]
    )
    undo_select.click(
        undo_mask_selection,
        [seg_all_res, clicked_points],
        [source_image_click, clicked_points, click_mask],
        show_progress=True
    )


    remove_phi.click(
        process_clean_phi,
        [origin_image, txt_detailed, txt_resegement],
        [img_rm_with_mask, click_mask, source_image_click],
        show_progress=True
    )
    save_clean_phi.click(
        image_upload,
        [img_rm_with_mask, image_resolution],
        outputs=[clicked_points, origin_image, seg_all_res,
                 features, orig_h, orig_w, input_h, input_w],
        show_progress=True, queue=True
    ).then(
        lambda origin_image: origin_image,
        [origin_image], [source_image_click]
    )
    inpaint_button.click(
        lambda: gr.Tabs(selected='removed'),
        outputs=[image_tab]
    ).then(
        get_inpainted_img,
        [origin_image, click_mask, image_resolution, sd_inference_step],
        [img_rm_with_mask]
    )


    clear_button_image.click(
        lambda origin_image, *reset_none: [[], origin_image] + [None] * len(reset_none),
        [origin_image, click_mask, img_rm_with_mask, seg_all_res],
        [clicked_points, source_image_click, click_mask,
         img_rm_with_mask, seg_all_res]
    )


if __name__ == "__main__":
    demo.queue(api_open=False).launch(
        server_name='0.0.0.0',
        share=True,
        debug=True,
    )