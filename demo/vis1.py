import numpy as np
import cv2
import sys
def render_mesh_pytorch3d(img, vertices, faces, cam_param):
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PerspectiveCameras, look_at_view_transform,
        RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftPhongShader, TexturesVertex, DirectionalLights
    )
    
    # # 转换为 PyTorch 张量
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    # faces = torch.tensor(faces, dtype=torch.int64, device=device)
    
    # # 设置相机参数
    # focal, princpt = cam_param['focal'], cam_param['princpt']
    # image_size = torch.tensor([img.shape[0], img.shape[1]], device=device)
    
    # # 计算相机参数
    # fx, fy = focal[0], focal[1]
    # cx, cy = princpt[0], princpt[1]
    
    # # 创建相机
    # cameras = PerspectiveCameras(
    #     focal_length=torch.tensor([[fx, fy]], device=device),
    #     principal_point=torch.tensor([[cx, cy]], device=device),
    #     image_size=((img.shape[0], img.shape[1]),), 
    #     device=device
    # )
    
    # # 设置纹理颜色
    # verts_rgb = torch.ones_like(vertices)[None]  # (1, V, 3)
    # verts_rgb[0, :, 0] = 0.7  # R
    # verts_rgb[0, :, 1] = 0.7  # G
    # verts_rgb[0, :, 2] = 0.7  # B
    # textures = TexturesVertex(verts_features=verts_rgb)
    
    # # 创建网格
    # mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
    
    # # 设置渲染器
    # raster_settings = RasterizationSettings(
    #     image_size=(img.shape[0], img.shape[1]),
    #     blur_radius=0.0,
    #     faces_per_pixel=1,
    # )
    
    # renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    #     shader=SoftPhongShader(device=device, cameras=cameras)
    # )
    
    # # 渲染
    # rendered_img = renderer(mesh)
    # rendered_img = rendered_img[0, ..., :3].cpu().numpy()
    
    # # 混合原始图像和渲染图像
    # alpha = 0.8
    # img = img / 255.0
    # output_img = rendered_img * alpha + img * (1 - alpha)
    # img = (output_img * 255).astype(np.uint8)
    
    # return img


    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 转换顶点和面
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    
    # 修正顶点Y坐标以解决上下颠倒问题
    # 方法1：直接翻转Y坐标 (假设Y是第二个坐标轴)
    vertices_fixed = vertices.clone()
    vertices_fixed[:, 0] = -vertices_fixed[:, 0]  # 翻转X坐标
    vertices_fixed[:, 1] = -vertices_fixed[:, 1]  # 翻转Y坐标
    
    # 提取相机参数
    focal = torch.tensor(cam_param['focal'], dtype=torch.float32, device=device)
    princpt = torch.tensor(cam_param['princpt'], dtype=torch.float32, device=device)
    
    fx, fy = focal[0].item(), focal[1].item()
    cx, cy = princpt[0].item(), princpt[1].item()
    
    # 图像尺寸
    img_h, img_w = img.shape[0], img.shape[1]
    
    # 创建相机 - 使用更精确的参数
    cameras = PerspectiveCameras(
        focal_length=[[fx, fy]],
        principal_point=[[cx, cy]],
        in_ndc=False,
        image_size=[[img_h, img_w]],
        device=device
    )
    
    # 增强光照设置，提高渲染质量
    lights = DirectionalLights(
        device=device,
        direction=[[0.0, 0.0, -1.0]],  # 光线方向
        ambient_color=[[0.4, 0.4, 0.4]],  # 环境光
        diffuse_color=[[0.6, 0.6, 0.6]],  # 漫反射
        specular_color=[[0.3, 0.3, 0.3]]   # 高光
    )
    
    # 创建更好的纹理
    verts_rgb = torch.ones_like(vertices_fixed)[None]  # (1, V, 3)
    verts_rgb[0, :, 0] = 0.7  # R
    verts_rgb[0, :, 1] = 0.7  # G
    verts_rgb[0, :, 2] = 0.7  # B
    textures = TexturesVertex(verts_features=verts_rgb)
    
    # 创建网格 - 使用修正后的顶点
    mesh = Meshes(verts=[vertices_fixed], faces=[faces], textures=textures)
    
    # 提高光栅化精度
    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w),
        blur_radius=0.0,
        faces_per_pixel=5,  # 增加每个像素的面数
        perspective_correct=True,  # 启用透视校正
        clip_barycentric_coords=True,
        cull_backfaces=True  # 剔除背面
    )
    
    # 创建渲染器 - 使用增强的光照
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )
    
    # 渲染
    rendered_img = renderer(mesh).cpu().numpy()
    alpha = 1
    valid_mask = (rendered_img[0, ..., 3] > 0) * alpha
    rendered_img = rendered_img[0, ..., :3]
    
    # 如果仍然上下颠倒，可以在这里翻转图像
    # rendered_img = rendered_img[::-1, :, :]  # 上下翻转
    
    # # 改进混合方式
    # alpha = 0.8
    # img_float = img.astype(np.float32) / 255.0
    
    # # 使用更好的混合方式
    # output_img = rendered_img * alpha + img_float * (1 - alpha)

    
    # 混合原始图像和渲染图像
    
    img = img / 255.0
    # rendered_img = rendered_img[:, :, :3] / 255.0
    valid_mask = valid_mask.astype(np.float32)
    valid_mask = np.stack([valid_mask] * 3, axis=-1)
    output_img = rendered_img * valid_mask + img * (1 - valid_mask)
    
    # 确保输出在合理范围内
    output_img = np.clip(output_img, 0, 1)
    result = (output_img * 255).astype(np.uint8)
    
    return result

# 2. 使用 Matplotlib
# Matplotlib 可以进行简单的3D渲染，不需要 OpenGL：
def render_mesh_matplotlib(img, vertices, faces, cam_param):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    # 创建图形
    fig = plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # 渲染网格
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                    triangles=faces, color=(0.7, 0.7, 0.7), alpha=0.8)
    
    # 设置相机视角
    focal, princpt = cam_param['focal'], cam_param['princpt']
    ax.view_init(elev=0, azim=180)
    ax.set_xlim([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
    ax.set_ylim([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
    ax.set_zlim([np.min(vertices[:, 2]), np.max(vertices[:, 2])])
    
    # 移除轴和背景
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    
    # 渲染到图像
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rendered_img = np.array(canvas.buffer_rgba())
    plt.close(fig)
    
    # 调整大小以匹配原始图像
    rendered_img = cv2.resize(rendered_img, (img.shape[1], img.shape[0]))
    
    # 混合原始图像和渲染图像
    alpha = 0.8
    mask = (rendered_img[:, :, 3:4] > 0) * alpha
    img = img / 255.0
    rendered_img = rendered_img[:, :, :3] / 255.0
    output_img = rendered_img * mask + img * (1 - mask)
    
    return (output_img * 255).astype(np.uint8)

# 3. 使用 Trimesh 的内置渲染器
def render_mesh_trimesh(img, vertices, faces, cam_param):
    import trimesh
    from trimesh.visual import ColorVisuals
    
    # 创建网格
    colors = np.ones((len(vertices), 4)) * [0.7, 0.7, 0.7, 1.0]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
                          visual=ColorVisuals(vertex_colors=colors))
    
    # 设置相机
    focal, princpt = cam_param['focal'], cam_param['princpt']
    resolution = (img.shape[0], img.shape[1])
    
    # 渲染场景
    scene = trimesh.Scene(mesh)
    
    # 使用简单的光栅化渲染器
    rendered_img = scene.save_image(resolution=resolution, 
                                 visible=True)
    
    # 转换为 numpy 数组
    rendered_img = np.array(rendered_img)
    
    # 混合原始图像和渲染图像
    alpha = 0.8
    valid_mask = (rendered_img[:, :, 3:4] > 0) * alpha
    img = img / 255.0
    rendered_img = rendered_img[:, :, :3] / 255.0
    output_img = rendered_img * valid_mask + img * (1 - valid_mask)
    
    return (output_img * 255).astype(np.uint8)

def main():
    vis_mesh = np.load("vis_mesh.npy")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"vis_mesh shape: {vis_mesh.shape}, dtype: {vis_mesh.dtype}")
    print(f"vis_mesh min: {vis_mesh.min()}, max: {vis_mesh.max()}")
    print(f"vis_mesh is contiguous: {vis_mesh.flags['C_CONTIGUOUS']}")
    # mesh, faces, cam_param = np.load("mesh.npy"), np.load("face.npy"), np.load("cam.npy")
    # cam_param = {'focal': cam_param[0], 'princpt': cam_param[1]}
    # vis_mesh = render_mesh_pytorch3d(vis_mesh, mesh, faces, cam_param)
    # print(vis_mesh.shape)
    cv2.imwrite("vis_mesh_pytorch3d.jpg", vis_mesh[:,:,::-1])

if __name__ == '__main__':
    main()