import torch
import sys
import os.path as osp
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', "common"))
from model_core import get_model

def load_and_analyze_weights(model, checkpoint_path):
    print(f"\n>>> 正在加载权重: {checkpoint_path}")
    
    # 1. 加载 Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_state_dict = model.state_dict()
    new_state_dict = {}
    
    loaded_keys = []
    missing_keys = []
    shape_mismatch_keys = []
    
    # 2. 遍历模型当前的参数
    for k, v in model_state_dict.items():
        # 尝试几种常见的前缀匹配逻辑
        possible_keys = [
            k,                      # 完全匹配
            f"module.{k}",          # DDP 前缀
            f"backbone.{k}",        # Backbone 前缀
            k.replace("backbone.", "") # 去掉 Backbone 前缀
        ]
        
        found = False
        for pk in possible_keys:
            if pk in state_dict:
                param = state_dict[pk]
                # 检查形状是否一致
                if param.shape == v.shape:
                    new_state_dict[k] = param
                    loaded_keys.append(k)
                    found = True
                    break
                else:
                    shape_mismatch_keys.append(f"{k} (Model: {v.shape} vs Ckpt: {param.shape})")
                    found = True # 找到了 key 但形状不对，暂且算作 found 以免归入 missing
                    break
        
        if not found:
            missing_keys.append(k)

    # 3. 加载匹配好的权重
    model.load_state_dict(new_state_dict, strict=False)
    
    # 4. 打印详细报告
    print("\n" + "="*50)
    print(f"权重加载报告")
    print("="*50)
    print(f"成功加载: {len(loaded_keys)} / {len(model_state_dict)} 层")
    
    if shape_mismatch_keys:
        print(f"\n[!!!] 形状不匹配 (严重警告，这些层未加载):")
        for k in shape_mismatch_keys:
            print(f"  - {k}")
            
    if missing_keys:
        print(f"\n[!] 未在权重文件中找到 (如果这是预训练加载，Head 部分缺失是正常的):")
        # 只打印前20个，避免刷屏
        for k in missing_keys[:20]:
            print(f"  - {k}")
        if len(missing_keys) > 20:
            print(f"  ... 以及其他 {len(missing_keys)-20} 个")

    # 5. 特别检查 Attention 部分
    print("\n[关键检查] MSDeformAttn 参数状态:")
    attn_keys = [k for k in model_state_dict.keys() if "sampling_offsets" in k]
    if not attn_keys:
        print("  未在模型中发现 'sampling_offsets'，可能层命名不同。")
    else:
        all_loaded = True
        for k in attn_keys:
            if k in loaded_keys:
                print(f"  [OK] {k}")
            else:
                print(f"  [MISSING] {k}")
                all_loaded = False
        if all_loaded:
            print("  >>> 恭喜！自定义的 PyTorch Attention 权重匹配成功。")
        else:
            print("  >>> 警告：Attention 权重未完全加载，请检查命名！")
    
    print("="*50 + "\n")

#使用方法：
model = get_model('test') # 或者 'test'
load_and_analyze_weights(model, "/workspace/OSX/pretrained_models/osx_l.pth.tar")