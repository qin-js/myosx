import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from einops import repeat


def denormalize_image(x, mean, std):
    """
    x: [1,3,H,W]
    mean/std: list or tuple of len=3
    return: [H,W,3] numpy in [0,1]
    """
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    y = x * std + mean
    y = y.clamp(0, 1)
    return y[0].permute(1, 2, 0).detach().cpu().numpy()


def save_rgb_image(arr, path):
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def upsample_2d(x, out_h, out_w, mode="bilinear"):
    t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        t = F.interpolate(t, size=(out_h, out_w), mode=mode)
    else:
        t = F.interpolate(t, size=(out_h, out_w), mode=mode, align_corners=False)
    return t[0, 0].cpu().numpy()


def upsample_3d(x, out_h, out_w, mode="bilinear"):
    t = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    if mode == "nearest":
        t = F.interpolate(t, size=(out_h, out_w), mode=mode)
    else:
        t = F.interpolate(t, size=(out_h, out_w), mode=mode, align_corners=False)
    return t[0].permute(1, 2, 0).cpu().numpy()


def pca_vis(feat_map):
    """
    feat_map: torch.Tensor [C,H,W]
    return: [H,W,3] numpy in [0,1]
    """
    C, H, W = feat_map.shape
    x = feat_map.reshape(C, H * W).transpose(1, 0).detach().cpu().numpy()  # [HW, C]

    pca = PCA(n_components=3)
    y = pca.fit_transform(x)

    y = y - y.min(axis=0, keepdims=True)
    denom = y.max(axis=0, keepdims=True) - y.min(axis=0, keepdims=True)
    denom[denom < 1e-8] = 1.0
    y = y / denom

    return y.reshape(H, W, 3)


def mean_act_vis(feat_map):
    """
    feat_map: [C,H,W]
    return: [H,W] numpy in [0,1]
    """
    act = feat_map.mean(dim=0)
    act = act - act.min()
    if act.max() > 1e-8:
        act = act / act.max()
    return act.detach().cpu().numpy()


def save_overlay(base_img, heatmap, path, cmap="jet", alpha=0.45, dpi=220):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(base_img)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_pca_overlay(base_img, pca_img, path, alpha=0.55, dpi=220):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(base_img)
    plt.imshow(pca_img, alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def make_summary_grid(input_img, items, save_path, mode="pca", dpi=220):
    n = len(items)
    fig = plt.figure(figsize=(4 * (n + 1), 4))

    plt.subplot(1, n + 1, 1)
    plt.imshow(input_img)
    plt.title("Input")
    plt.axis("off")

    for idx, item in enumerate(items, start=2):
        plt.subplot(1, n + 1, idx)
        if mode == "pca":
            plt.imshow(item["vis"])
            plt.title(f"L{item['layer_num']} PCA")
        elif mode == "meanact":
            plt.imshow(input_img)
            plt.imshow(item["vis"], cmap="jet", alpha=0.45)
            plt.title(f"L{item['layer_num']} MeanAct")
        else:
            raise ValueError(mode)
        plt.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def collect_encoder_intermediates(
    encoder,
    x,
    out_indices,
    collect_attention=False,
    attention_head_fusion="mean",
    attention_task_token_indices=(0,),
):
    """
    Returns:
        intermediates: list of dict
            {
                "layer_idx": int,
                "layer_num": int,
                "feature_map": [B,C,Hp,Wp],
                "hw": (Hp,Wp),

                # collect_attention=True 时附加：
                "attn": [B, heads, N, N],
                "task_to_patch": dict(task_idx -> [B, heads, Hp, Wp]),
                "task_to_patch_fused": dict(task_idx -> [B, Hp, Wp]),
            }
    """
    B, C, H, W = x.shape

    with torch.no_grad():
        x, (Hp, Wp) = encoder.patch_embed(x)

        task_tokens = repeat(encoder.task_tokens, '() n d -> b n d', b=B)

        if encoder.pos_embed is not None:
            x = x + encoder.pos_embed[:, 1:] + encoder.pos_embed[:, :1]

        x = torch.cat((task_tokens, x), dim=1)

        intermediates = []

        for i, blk in enumerate(encoder.blocks):
            if collect_attention:
                x, attn = forward_block_with_attention(blk, x)
            else:
                x = blk(x)
                attn = None

            if i in out_indices:
                patch_tokens = x[:, encoder.task_tokens_num:]  # [B, Hp*Wp, C]
                feat = patch_tokens.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

                item = {
                    "layer_idx": i,
                    "layer_num": i + 1,
                    "feature_map": feat.detach().cpu(),
                    "hw": (Hp, Wp),
                }

                if collect_attention and attn is not None:
                    # attn: [B, heads, N, N]
                    # 前 task_tokens_num 个 token 是 task tokens
                    task_to_patch = {}
                    task_to_patch_fused = {}

                    for task_idx in attention_task_token_indices:
                        if task_idx >= encoder.task_tokens_num:
                            continue

                        # 第 task_idx 个 task token 对所有 patch token 的注意力
                        # token 索引：task token 在前面，所以源 token 位置就是 task_idx
                        # 目标 patch token 范围是 encoder.task_tokens_num:
                        t2p = attn[:, :, task_idx, encoder.task_tokens_num:]  # [B, heads, Hp*Wp]
                        t2p = t2p.reshape(B, blk.attn.num_heads, Hp, Wp)

                        if attention_head_fusion == "mean":
                            fused = t2p.mean(dim=1)
                        elif attention_head_fusion == "max":
                            fused = t2p.max(dim=1).values
                        else:
                            raise ValueError(f"Unsupported attention_head_fusion: {attention_head_fusion}")

                        task_to_patch[task_idx] = t2p.detach().cpu()
                        task_to_patch_fused[task_idx] = fused.detach().cpu()

                    item["attn"] = attn.detach().cpu()
                    item["task_to_patch"] = task_to_patch
                    item["task_to_patch_fused"] = task_to_patch_fused

                intermediates.append(item)

    return intermediates

def normalize_map(x):
    x = x - x.min()
    if x.max() > 1e-8:
        x = x / x.max()
    return x

def save_heatmap_only(heatmap, path, cmap="jet", dpi=220):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def make_attention_summary_grid(input_img, items, save_path, title_prefix="TaskAttn", dpi=220):
    n = len(items)
    fig = plt.figure(figsize=(4 * (n + 1), 4))

    plt.subplot(1, n + 1, 1)
    plt.imshow(input_img)
    plt.title("Input")
    plt.axis("off")

    for idx, item in enumerate(items, start=2):
        plt.subplot(1, n + 1, idx)
        plt.imshow(input_img)
        plt.imshow(item["vis"], cmap="jet", alpha=0.45)
        plt.title(f"L{item['layer_num']} {title_prefix}")
        plt.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def forward_block_with_attention(blk, x):
    """
    手动展开一个 transformer block，返回:
        x_out: block 输出
        attn: [B, heads, N, N]
    """
    # norm1
    x_norm = blk.norm1(x)  # [B, N, C]
    B, N, C = x_norm.shape

    # qkv
    qkv = blk.attn.qkv(x_norm)  # [B, N, 3*all_head_dim]
    qkv = qkv.reshape(B, N, 3, blk.attn.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

    q = q * blk.attn.scale
    attn = q @ k.transpose(-2, -1)  # [B, heads, N, N]
    attn = attn.softmax(dim=-1)
    attn = blk.attn.attn_drop(attn)

    # attn output
    attn_out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    attn_out = blk.attn.proj(attn_out)
    attn_out = blk.attn.proj_drop(attn_out)

    # residual + mlp
    x = x + blk.drop_path(attn_out)
    x = x + blk.drop_path(blk.mlp(blk.norm2(x)))

    return x, attn


def save_encoder_visualizations(
    encoder,
    input_tensor,
    vis_cfg,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    image_name="sample",
):
    if not vis_cfg.get("enable", False):
        return

    save_dir = Path(vis_cfg.get("save_dir", "work_dirs/vit_vis")) / image_name
    save_dir.mkdir(parents=True, exist_ok=True)

    out_indices = set(vis_cfg.get("layers", [3, 6, 9, 12, 15, 18, 21, 23]))
    upsample_mode = vis_cfg.get("upsample_mode", "bilinear")
    single_dpi = vis_cfg.get("single_dpi", 240)
    summary_dpi = vis_cfg.get("summary_dpi", 220)

    save_input = vis_cfg.get("save_input", True)
    save_pca = vis_cfg.get("save_pca", True)
    save_meanact = vis_cfg.get("save_meanact", True)
    save_overlay_flag = vis_cfg.get("save_overlay", True)

    # 新增 attention 配置
    save_attention = vis_cfg.get("save_attention", False)
    attention_head_fusion = vis_cfg.get("attention_head_fusion", "mean")
    attention_task_token_indices = tuple(vis_cfg.get("attention_task_token_indices", [0]))
    save_attention_per_head = vis_cfg.get("save_attention_per_head", False)

    input_img = denormalize_image(input_tensor, img_mean, img_std)
    H, W = input_img.shape[:2]

    if save_input:
        save_rgb_image(input_img, save_dir / "input.png")

    intermediates = collect_encoder_intermediates(
        encoder=encoder,
        x=input_tensor,
        out_indices=out_indices,
        collect_attention=save_attention,
        attention_head_fusion=attention_head_fusion,
        attention_task_token_indices=attention_task_token_indices,
    )

    pca_items = []
    meanact_items = []
    attention_summary_items = {task_idx: [] for task_idx in attention_task_token_indices}

    for item in intermediates:
        layer_num = item["layer_num"]
        feat = item["feature_map"][0]  # [C,h,w]
        _, h, w = feat.shape

        # ---------- PCA ----------
        if save_pca:
            pca_raw = pca_vis(feat)
            pca_up = upsample_3d(pca_raw, H, W, mode=upsample_mode)

            save_rgb_image(pca_raw, save_dir / f"layer_{layer_num:02d}_pca_raw_{h}x{w}.png")
            save_rgb_image(pca_up, save_dir / f"layer_{layer_num:02d}_pca_up.png")

            if save_overlay_flag:
                save_pca_overlay(
                    input_img, pca_up,
                    save_dir / f"layer_{layer_num:02d}_pca_overlay.png",
                    alpha=0.55,
                    dpi=single_dpi
                )

            pca_items.append({
                "layer_num": layer_num,
                "vis": pca_up
            })

        # ---------- Mean Activation ----------
        if save_meanact:
            act_raw = mean_act_vis(feat)
            act_up = upsample_2d(act_raw, H, W, mode=upsample_mode)

            act_raw_rgb = np.stack([act_raw] * 3, axis=-1)
            save_rgb_image(act_raw_rgb, save_dir / f"layer_{layer_num:02d}_meanact_raw_{h}x{w}.png")

            if save_overlay_flag:
                save_overlay(
                    input_img, act_up,
                    save_dir / f"layer_{layer_num:02d}_meanact_overlay.png",
                    cmap="jet",
                    alpha=0.45,
                    dpi=single_dpi
                )

            meanact_items.append({
                "layer_num": layer_num,
                "vis": act_up
            })

        # ---------- Attention ----------
        if save_attention and "task_to_patch_fused" in item:
            for task_idx, attn_map_tensor in item["task_to_patch_fused"].items():
                # [B, Hp, Wp] -> 只保存第一张
                attn_raw = attn_map_tensor[0].numpy()
                attn_raw = normalize_map(attn_raw)
                attn_up = upsample_2d(attn_raw, H, W, mode=upsample_mode)
                attn_up = normalize_map(attn_up)

                # raw heatmap
                attn_raw_rgb = np.stack([attn_raw] * 3, axis=-1)
                save_rgb_image(
                    attn_raw_rgb,
                    save_dir / f"layer_{layer_num:02d}_task{task_idx}_attn_raw_{h}x{w}.png"
                )

                # overlay
                if save_overlay_flag:
                    save_overlay(
                        input_img,
                        attn_up,
                        save_dir / f"layer_{layer_num:02d}_task{task_idx}_attn_overlay.png",
                        cmap="jet",
                        alpha=0.45,
                        dpi=single_dpi
                    )

                attention_summary_items[task_idx].append({
                    "layer_num": layer_num,
                    "vis": attn_up
                })

                # 是否保存每个 head
                if save_attention_per_head and "task_to_patch" in item:
                    per_head = item["task_to_patch"][task_idx][0]  # [heads, Hp, Wp]
                    num_heads = per_head.shape[0]

                    for head_idx in range(num_heads):
                        head_map = per_head[head_idx].numpy()
                        head_map = normalize_map(head_map)
                        head_up = upsample_2d(head_map, H, W, mode=upsample_mode)
                        head_up = normalize_map(head_up)

                        save_overlay(
                            input_img,
                            head_up,
                            save_dir / f"layer_{layer_num:02d}_task{task_idx}_head{head_idx:02d}_attn_overlay.png",
                            cmap="jet",
                            alpha=0.45,
                            dpi=single_dpi
                        )

    # ---------- summary ----------
    if len(pca_items) > 0:
        make_summary_grid(
            input_img=input_img,
            items=pca_items,
            save_path=save_dir / "pca_summary.png",
            mode="pca",
            dpi=summary_dpi
        )

    if len(meanact_items) > 0:
        make_summary_grid(
            input_img=input_img,
            items=meanact_items,
            save_path=save_dir / "meanact_summary.png",
            mode="meanact",
            dpi=summary_dpi
        )

    if save_attention:
        for task_idx, items in attention_summary_items.items():
            if len(items) > 0:
                make_attention_summary_grid(
                    input_img=input_img,
                    items=items,
                    save_path=save_dir / f"task{task_idx}_attention_summary.png",
                    title_prefix=f"T{task_idx}Attn",
                    dpi=summary_dpi
                )