import torch
import os

src = "/home/liushuai/project/fusion_vp/MI-DETR/output/5090/trans/model_0055999.pth"          # 原始大 ckpt
dst = "/home/liushuai/project/fusion_vp/MI-DETR/output/5090/trans/model_0055999shark16.pth"  # 压缩后的 ckpt

# 1. 加载原始 checkpoint
ckpt = torch.load(src, map_location="cpu")
print("ckpt keys:", ckpt.keys())

# 2. 只取模型权重部分
if "model" in ckpt:
    state = ckpt["model"]
elif "model_state" in ckpt:
    state = ckpt["model_state"]
else:
    # 有些项目直接就是 state_dict
    state = ckpt

# 3. 把所有 float32 权重压成 float16
for k, v in list(state.items()):
    if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
        if v.dtype == torch.float32:
            state[k] = v.half()

# 4. 只保存模型（fp16）权重
torch.save({"model": state}, dst)

print("old size: {:.2f} MB".format(os.path.getsize(src) / 1024**2))
print("new size: {:.2f} MB".format(os.path.getsize(dst) / 1024**2))