import os
import torch
import numpy as np
from PIL import Image

def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5

def transform_point_4x4(p, matrix):
    p_hom = np.dot(matrix, np.append(p, 1.0))
    p_w = 1.0 / (p_hom[3] + 1e-7)
    p_proj = p_hom[:3] * p_w
    return p_proj

def transform_point_4x3(p, matrix):
    p_cam = np.dot(matrix, np.append(p, 1.0))
    return p_cam

def draw_point(image, x, y, color, point_size=3):
    half_size = point_size // 2
    for i in range(-half_size, half_size + 1):
        for j in range(-half_size, half_size + 1):
            if 0 <= x + i < image.shape[1] and 0 <= y + j < image.shape[0]:
                image[y + j, x + i] = color

def save_points_projected_view(init_pcd, proj_matrix, view_matrix, image_width, image_height, output_image_path):
    points = init_pcd.points
    colors = init_pcd.colors
    
    # Move the projection matrix to CPU and convert to numpy
    view_matrix_cpu = view_matrix.cpu().numpy().T
    proj_matrix_cpu = proj_matrix.cpu().numpy().T
    
    # Prepare image (with RGB channels)
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    
    for idx, point in enumerate(points):
        p_cam = transform_point_4x3(point, view_matrix_cpu)
        if p_cam[2] < 0.2:  # 深度过滤
            continue
        p_proj = transform_point_4x4(point, proj_matrix_cpu)
        x_img = int(ndc2Pix(p_proj[0], image_width))
        y_img = int(ndc2Pix(p_proj[1], image_height))
        
        if 0 <= x_img < image_width and 0 <= y_img < image_height:
            color = (colors[idx] * 255).astype(np.uint8)
            draw_point(image, x_img, y_img, color, point_size=2)
    
    pil_image = Image.fromarray(image, 'RGB')
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    pil_image.save(output_image_path, format='JPEG')
    print(f"point image view saved to {output_image_path}")

def save_tensor_3NN_as_image(tensor, filename):
    """
    将 PyTorch 张量保存为 JPG 图像并显示。
    
    参数:
    tensor (torch.Tensor): 形状为 (3, height, width) 的 PyTorch 张量。
    filename (str): 保存图像的文件名（包括路径和 .jpg 后缀）。
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 将张量从 GPU 转回 CPU，并移除梯度信息，然后转换为 numpy 数组
    array = tensor.detach().cpu().numpy()
    
    # 调整维度顺序从 (3, height, width) 到 (height, width, 3)
    array = np.transpose(array, (1, 2, 0))
    
    # 将归一化的数组转换到 [0, 255] 范围
    array = (array * 255).astype(np.uint8)
    
    # 将 numpy 数组转换为 PIL 图像
    image = Image.fromarray(array)
    
    # 保存 PIL 图像为 JPG 格式
    image.save(filename, format='JPEG')