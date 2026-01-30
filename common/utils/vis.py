import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"

# import pyrender
import trimesh
from config import cfg

def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
    y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)
    
    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

# def render_mesh(img, mesh, face, cam_param):
#     # mesh
#     mesh = trimesh.Trimesh(mesh, face)
#     rot = trimesh.transformations.rotation_matrix(
# 	np.radians(180), [1, 0, 0])
#     mesh.apply_transform(rot)
#     material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
#     mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
#     scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
#     scene.add(mesh, 'mesh')
    
#     focal, princpt = cam_param['focal'], cam_param['princpt']
#     camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
#     scene.add(camera)
 
#     # renderer
#     renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
#     # light
#     light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
#     light_pose = np.eye(4)
#     light_pose[:3, 3] = np.array([0, -1, 1])
#     scene.add(light, pose=light_pose)
#     light_pose[:3, 3] = np.array([0, 1, 1])
#     scene.add(light, pose=light_pose)
#     light_pose[:3, 3] = np.array([1, 1, 2])
#     scene.add(light, pose=light_pose)

#     # render
#     rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
#     rgb = rgb[:,:,:3].astype(np.float32)
#     valid_mask = (depth > 0)[:,:,None]

#     # save to image
#     img = rgb * valid_mask + img * (1-valid_mask)
#     return img

def render_mesh(img, vertices, faces, cam_param, mesh_as_vertices=False):
    if mesh_as_vertices:
        # to run on cluster where headless pyrender is not supported for A100/V100
        vertices_2d = perspective_projection(vertices, cam_param)
        img = vis_keypoints(img, vertices_2d, alpha=0.8, radius=2, color=(0, 0, 255))
        return img
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PerspectiveCameras, look_at_view_transform,
        RasterizationSettings, MeshRenderer, MeshRasterizer,
        SoftPhongShader, TexturesVertex, DirectionalLights
    )
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 转换顶点和面
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(faces.astype(np.int64), dtype=torch.int64, device=device)
    
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