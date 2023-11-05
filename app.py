import gradio as gr
import torch
from matplotlib import pyplot as plt
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import SamPredictor, sam_model_registry
from torchvision.ops import box_convert

model_type = "vit_b"
sam_checkpoint = "weights/sam_vit_b.pth"
config = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
dino_checkpoint = "weights/groundingdino_swint_ogc.pth"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
device = "cpu"
model = load_model(config, dino_checkpoint, device)
box_threshold = 0.35
text_threshold = 0.25

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label = None):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))  
    if label is not None:
        ax.text(x0, y0, label, fontsize=12, color='white', backgroundcolor='red', ha='left', va='top')

def extract_object_with_transparent_background(image, masks):
    mask_expanded = np.expand_dims(masks[0], axis=-1)
    mask_expanded = np.repeat(mask_expanded, 3, axis=-1)
    segment = image * mask_expanded
    rgba_segment = np.zeros((segment.shape[0], segment.shape[1], 4), dtype=np.uint8)
    rgba_segment[:, :, :3] = segment
    rgba_segment[:, :, 3] = masks[0] * 255
    return rgba_segment

def extract_remaining_image(image, masks):
    inverse_mask = np.logical_not(masks[0])
    inverse_mask_expanded = np.expand_dims(inverse_mask, axis=-1)
    inverse_mask_expanded = np.repeat(inverse_mask_expanded, 3, axis=-1)
    remaining_image = image * inverse_mask_expanded
    return remaining_image

def overlay_masks_boxes_on_image(image, masks, boxes, labels, show_masks, show_boxes):
    fig, ax = plt.subplots()
    ax.imshow(image)
    if show_masks:
        for mask in masks:
            show_mask(mask, ax, random_color=False)

    if show_boxes:
        for input_box, label in zip(boxes, labels):
            show_box(input_box, ax, label)

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.margins(0, 0)

    fig.canvas.draw() 
    output_image = np.array(fig.canvas.buffer_rgba())
    
    plt.close(fig)
    return output_image


def detect_objects(image, prompt, show_masks, show_boxes, crop_options):
    image_source, image = load_image(image)
    predictor.set_image(image_source)
    
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=prompt, 
        box_threshold=box_threshold, 
        text_threshold=text_threshold,
        device=device
    )

    h, w, _ = image_source.shape
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy") * torch.Tensor([w, h, w, h])
    boxes = np.round(boxes.numpy()).astype(int)

    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]

    masks_list = []

    for input_box, label in zip(boxes, labels):
        x1, y1, x2, y2 = input_box
        width = x2 - x1
        height = y2 - y1
        avg_size = (width + height) / 2
        d = avg_size * 0.1 
        
        center_point = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        points = []
        points.append([center_point[0], center_point[1] - d]) 
        points.append([center_point[0], center_point[1] + d])  
        points.append([center_point[0] - d, center_point[1]]) 
        points.append([center_point[0] + d, center_point[1]])  
        input_point = np.array(points)
        input_label = np.array([1] * len(input_point))

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask_input = logits[np.argmax(scores), :, :]

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False
        )
        masks_list.append(masks)
        
    if crop_options == "Crop":
        composite_image = np.zeros_like(image_source)
        for masks in masks_list:
            rgba_segment = extract_object_with_transparent_background(image_source, masks)
            composite_image = np.maximum(composite_image, rgba_segment[:, :, :3])
        output_image = overlay_masks_boxes_on_image(composite_image, masks_list, boxes, labels, show_masks, show_boxes)
    elif crop_options == "Inverse Crop":
        remaining_image = image_source.copy()
        for masks in masks_list:
            remaining_image = extract_remaining_image(remaining_image, masks)
        output_image = overlay_masks_boxes_on_image(remaining_image, masks_list, boxes, labels, show_masks, show_boxes)
    else:
        output_image = overlay_masks_boxes_on_image(image_source, masks_list, boxes, labels, show_masks, show_boxes)
    
    output_image_path = 'output_image.jpeg'
    plt.imsave(output_image_path, output_image)
    
    return output_image_path

# block = gr.Blocks(css=".gradio-container {background-color: #f8f8f8; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif}" )
block = gr.Blocks(theme='JohnSmith9982/small_and_pretty', css="footer{display:none !important}")

with block:
    gr.HTML("""
    <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Roboto', sans-serif;"
            padding: 30px;
        }
    </style>
    """)
    
    gr.HTML("<h1 style='text-align: center;'>AI-Driven Image Annotation for Object Detection: A Hands-Free Approach</h1>")
    gr.HTML("<h3 style='text-align: center;'>This demo consists of a single file flow of our final project for MPI</h3>")
    with gr.Row():
        with gr.Column(width="auto"):
            input_image = gr.Image(type='filepath', label="Upload Image")
        with gr.Column(width="auto"):
            output_image = gr.Image(type='filepath', label="Result")
    with gr.Row():
        with gr.Column(width="auto"):
            object_search = gr.Textbox(
                label="Object to Detect",
                placeholder="Enter any text, comma separated if multiple objects needed",
                show_label=True,
                lines=1,
            )
        with gr.Column(width="auto"):
            show_masks = gr.Checkbox(label="Show Masks", default=True)
            show_boxes = gr.Checkbox(label="Show Boxes", default=True)
        with gr.Column(width="auto"):
            crop_options = gr.Radio(choices=["None", "Crop", "Inverse Crop"], label="Crop Options", default="None")
    with gr.Row():
        submit = gr.Button(value="Send", variant="secondary").style(full_width=True)

    gr.Examples(
        examples=[
            ["images/tiger.jpeg", "animal from cat family", True, True],
            ["images/car.jpeg", "a blue sports car", True, False],
            ["images/bags.jpeg", "black bag next to the red bag", False, True],
            ["images/deer.jpeg", "deer jumping and running across the road", True, True],
            ["images/penn.jpeg", "sign board", True, False],
        ],
        inputs=[input_image, object_search, show_masks, show_boxes],
    )
    gr.HTML("""
            <div style="text-align:center">
                <p>Developed by <a href='https://github.com/sharozhaseeb'>Sharoz Haseeb</a> and Owais Ali</p>
                <p>Powered by <a href='https://segment-anything.com'>Segment Anything</a> and <a href='https://arxiv.org/abs/2303.05499'>Grounding DINO</a></p>
                <p>Just upload an image and enter the objects to detect, segment, crop, etc.</p>
                <p>What's Zero-Shot? It means you can detect objects without any training samples!</p>
                
            </div>
            <style>
                p {
                    margin-bottom: 10px;
                    font-size: 16px;
                }
                a {
                    color: #3867d6;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
            """)

    submit.click(fn=detect_objects,
                inputs=[input_image, object_search, show_masks, show_boxes, crop_options],
                outputs=[output_image])

block.launch(width=800)