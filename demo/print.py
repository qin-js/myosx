import torch

# 替换成你的权重文件路径
pth_file = '../pretrained_models/osx_l.pth.tar' 
checkpoint = torch.load(pth_file, map_location='cpu')

if 'state_dict' in checkpoint:
    st = checkpoint['state_dict']
else:
    st = checkpoint

print("===== 权重文件 Key 示例 (前20个) =====")
for i, k in enumerate(st.keys()):
    if i >= 20: break
    print(k)

print("\n===== 查找 Attention 相关 Key =====")
for k in st.keys():
    if 'sampling_offsets' in k:
        print(k)
        break # 只看一个例子即可