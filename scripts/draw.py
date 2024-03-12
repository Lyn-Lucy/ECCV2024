import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def draw_attn_map(idx, prompt,image_tensor = None, attn_maps_dir='/home/lzh/llx/attn_map', output_dir='/home/lzh/llx/attn_map/vqa'):
    # 图片文件夹路径
    image_folder_path = f'/home/lzh/llx/ScienceQA/data/scienceqa/images/test/{idx}'
    # 创建输出文件夹，用于存放图片和文本
    idx_output_dir = os.path.join(output_dir, str(idx))
    os.makedirs(idx_output_dir, exist_ok=True) 

    # 保存prompt到txt文件
    prompt_path = os.path.join(idx_output_dir, 'prompt.txt')
    with open(prompt_path, 'w') as file:
        file.write(prompt)
    if image_tensor == None:
        # 将图片文件固定成正方形并保存
        for image_file in os.listdir(image_folder_path):
            image_path = os.path.join(image_folder_path, image_file)
            # 确保是文件并且是图片
            if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img = Image.open(image_path)
                # 转换图片为正方形
                img_square = crop_to_square(img)
                # 保存调整后的图片
                img_square.save(os.path.join(idx_output_dir, image_file))
    else:
        image_tensor = image_tensor.squeeze(0)
        if image_tensor.shape[0] == 3:
            # 使用 permute 来重排维度
            image_tensor = image_tensor.permute(1, 2, 0)
        # 使用 imshow 绘制图像
        plt.imshow(image_tensor)
        plt.axis('off')  # 不显示坐标轴
        plt.savefig(os.path.join(idx_output_dir, "image_file.png"), bbox_inches='tight', pad_inches=0)


    # 处理并保存attn_map
    for layer_idx in [0,9,19,29,39]:
        attn_map_file_path = os.path.join(attn_maps_dir, f'attn_map_layer{layer_idx}.pt')
        if os.path.isfile(attn_map_file_path):
            attn_map = torch.load(attn_map_file_path)
            if attn_map.shape[0] == 576:
                attn_map_reshaped = attn_map.view(24, 24).cpu().numpy()
                fig, ax = plt.subplots()
                cax = ax.imshow(attn_map_reshaped, cmap='gray', interpolation='none')
                fig.colorbar(cax)  # 这一行添加颜色条
                plt.axis('off')
                attn_map_image_path = os.path.join(idx_output_dir, f'attn_map_layer{layer_idx}.png')
                plt.savefig(attn_map_image_path, bbox_inches='tight', pad_inches=0)
                plt.close()
        else:
            print(f'Attention map {attn_map_file_path} does not exist.')

    similarity_path = os.path.join(attn_maps_dir, 'similarity.pt')
    similarity_map = torch.load(similarity_path)
    similarity_map_reshaped = similarity_map.view(24, 24).cpu().numpy()
    fig, ax = plt.subplots()
    cax = ax.imshow(similarity_map_reshaped, cmap='gray', interpolation='none')
    fig.colorbar(cax)  # 添加颜色条
    plt.axis('off')
    similarity_map_image_path = os.path.join(idx_output_dir, 'similarity.png')
    plt.savefig(similarity_map_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
            
def crop_to_square(image):
    width, height = image.size   # Get dimensions
    new_size = min(width, height)

    left = (width - new_size)/2
    top = (height - new_size)/2
    right = (width + new_size)/2
    bottom = (height + new_size)/2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    return image

