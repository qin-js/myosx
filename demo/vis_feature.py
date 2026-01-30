import torch
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 从文件中读取特征
features = []
for i in range(3, 25, 3):
    feature = torch.load(f"./features/f_{i}.pt")
    features.append(feature)


def apply_pca(feature, target_shape=(16, 12)):
    """
    使用 PCA 对特征进行降维。
    :param feature: 输入特征 (torch.Tensor)
    :param target_shape: 目标形状 (height, width)
    :return: 降维后的特征 (numpy.ndarray)
    """
    # 展平特征为二维数组 (H*W, C)
    original_shape = feature.shape
    flat_feature = feature.view(original_shape[0], original_shape[1], -1).permute(0, 2, 1)  # [1, H*W, C]
    flat_feature = flat_feature.squeeze(0).detach().cpu().numpy()  # [H*W, C]

    # 应用 PCA 降维到目标维度
    n_components = target_shape[0] * target_shape[1]  # 目标像素数
    pca = PCA(n_components=3)
    reduced_feature = pca.fit_transform(flat_feature)  # [H*W, n_components]

    # 重新调整为目标形状
    reshaped_feature = reduced_feature.reshape(target_shape)  # [H, W]
    return reshaped_feature

def save_feature_as_image(feature, output_path):
    """
    将单个特征保存为 PNG 图像。
    """
    # 归一化到 [0, 255] 范围
    feature = (feature - feature.min()) / (feature.max() - feature.min())
    print(feature.shape)
    feature = (feature * 255).astype(np.uint8)  # 转换为 8-bit 图像
    
    # 使用 Pillow 创建图像
    img = Image.fromarray(feature, mode="RGB")  # 单通道灰度图
    img.save(output_path)


def save_feature_as_colormap_image(feature, output_path, colormap='viridis'):
    """
    将单个特征保存为带有色彩映射的 PNG 图像。
    """
    # 归一化到 [0, 1] 范围
    feature = (feature - feature.min()) / (feature.max() - feature.min())
    
    # 使用 Matplotlib 的色彩映射
    cmap = plt.get_cmap(colormap)  # 获取色彩映射
    colored_feature = cmap(feature)[:, :, :3]  # 取 RGB 颜色通道，忽略 Alpha 通道
    colored_feature = (colored_feature * 255).astype(np.uint8)  # 转换为 8-bit 图像

    # 检查数组形状
    if colored_feature.ndim != 3 or colored_feature.shape[2] != 3:
        raise ValueError(f"Unexpected shape of colored_feature: {colored_feature.shape}. Expected (H, W, 3).")
    
    # 使用 Pillow 创建图像
    img = Image.fromarray(colored_feature, mode='RGB')  # RGB 图像
    img.save(output_path)

# def save_feature_as_image(feature, output_path):
#     """
#     将单个特征保存为 PNG 图像。
#     """
#     # 在通道维度上取平均值，压缩为单通道
#     mean_feature = feature.mean(dim=1).squeeze(0)  # 形状变为 [16, 12]
    
#     # 归一化到 [0, 255] 范围
#     mean_feature = (mean_feature - mean_feature.min()) / (mean_feature.max() - mean_feature.min())
#     mean_feature = (mean_feature * 255).byte()  # 转换为 8-bit 图像
    
#     # 转换为 NumPy 数组
#     image_array = mean_feature.detach().cpu().numpy()
    
#     # 使用 Pillow 创建图像
#     img = Image.fromarray(image_array, mode='L')  # 单通道灰度图
#     img.save(output_path)

def concatenate_images(image_paths, output_path, grid_size=None):
    """
    拼接多张图片为一张大图。
    :param image_paths: 图片路径列表
    :param output_path: 输出拼接图片的路径
    :param grid_size: 拼接网格大小 (rows, cols)，如果为 None，则自动计算
    """
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(img.size for img in images))
    
    if grid_size is None:
        grid_size = (1, len(images))  # 默认水平排列
    
    rows, cols = grid_size
    total_width = max(widths) * cols
    total_height = max(heights) * rows
    
    # 创建空白画布
    new_image = Image.new('RGB', (total_width, total_height))
    
    # 拼接图片
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        new_image.paste(img, (col * img.width, row * img.height))
    
    # 保存拼接后的图片
    new_image.save(output_path)

# 主程序
image_paths = []
output_dir = "./features/"  # 确保该目录存在

for i, feature in enumerate(features):
    reduced_feature = apply_pca(feature, target_shape=(16, 12, 3))
    output_path = f"{output_dir}feature_{i}.png"
    save_feature_as_image(reduced_feature, output_path)
    # save_feature_as_colormap_image(reduced_feature, output_path)
    image_paths.append(output_path)

# 拼接所有图片
concatenate_images(image_paths, f"{output_dir}concatenated_features.png", grid_size=(1, 8))