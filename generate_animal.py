#!/usr/bin/env python3
from PIL import Image
import torch
import os
import glob
import cv2
import shlex
import subprocess
import numpy as np
from torchvision import transforms
from transformers import AutoModelForImageSegmentation, AutoModel
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

text_prompt = "A old chinese woman is riding a blue mountain bicycle on highway. Side view."
print(text_prompt)

# Generate video of the animal
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    prompt=text_prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)

# Remove the background in the video
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
model.to('cuda')
model.eval()

image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

IMAGE_SIZE = 512
KERNEL_SIZE = 5
kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
os.makedirs('animal_images', exist_ok=True)
os.makedirs('animal_images_no_bg', exist_ok=True)
os.makedirs('animal_images_white_bg', exist_ok=True)
subprocess.run(shlex.split('ffmpeg -y -i output.mp4 -vf fps=8 ./animal_images/%2d.jpg'))
image_paths = sorted(glob.glob(os.path.join('animal_images', '*.jpg')))
for image_path in image_paths:
    print (image_path)
    image = Image.open(image_path)
    input_images = transform_image(image).unsqueeze(0).to('cuda')
    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    mask_cv = np.uint8(np.array(mask)) * 255
    mask_cv = cv2.erode(mask_cv, kernel)
    mask_x = np.sum(mask, axis=0)
    left = mask_x.nonzero()[0][0]
    right = mask_x.nonzero()[0][-1]
    mask_y = np.sum(mask, axis=1)
    up = mask_y.nonzero()[0][0]
    low = mask_y.nonzero()[0][-1]
    img_cv = np.array(image)
    img_cv = img_cv[:, :, ::-1].copy()
    img_3 = cv2.bitwise_and(img_cv, img_cv, mask=mask_cv)
    img_3[mask_cv==0] = (255,255,255)
    image.putalpha(mask)
    image.save(os.path.join('animal_images_no_bg', os.path.basename(image_path)[:-4]+'.png'))
    img_3 = img_3[up:low, left:right, :]
    H, W = img_3.shape[0], img_3.shape[1]
    scale = 450/np.maximum(H,W)
    img_3 = cv2.resize(img_3, (0, 0), fx=scale, fy=scale)
    H, W = img_3.shape[0], img_3.shape[1]
    top_lines = int((IMAGE_SIZE-H)/2)
    bottom_lines = IMAGE_SIZE - top_lines - H
    top_pad = np.ones((top_lines, W, 3), dtype=np.uint8) * 255
    bottom_pad = np.ones((bottom_lines, W, 3), dtype=np.uint8) * 255
    img_3 = np.concatenate((top_pad, img_3, bottom_pad), axis=0)
    left_lines = int((IMAGE_SIZE-W)/2)
    right_lines = IMAGE_SIZE - left_lines - W
    left_pad = np.ones((IMAGE_SIZE, left_lines, 3), dtype=np.uint8) * 255
    right_pad = np.ones((IMAGE_SIZE, right_lines, 3), dtype=np.uint8) * 255
    img_3 = np.concatenate((left_pad, img_3, right_pad), axis=1)
    assert img_3.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    cv2.imwrite(os.path.join('animal_images_white_bg', os.path.basename(image_path)[:-4]+'.jpg'), img_3)
    # break
subprocess.run(shlex.split('ffmpeg -y -framerate 8 -pattern_type glob -i "./animal_images_white_bg/*.jpg" -pix_fmt yuv420p animal_images_white_bg.mp4'))
subprocess.run(shlex.split('python infer_3d.py big --workspace results --resume pretrained/recon.safetensors --num_frames 1 --test_path animal_images_white_bg.mp4'))
subprocess.run(shlex.split('python infer_4d.py big --workspace results --resume pretrained/recon.safetensors --interpresume pretrained/interp.safetensors --num_frames 16 --test_path animal_images_white_bg.mp4'))
