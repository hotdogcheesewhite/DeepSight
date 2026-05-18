import torch
import sys
from safetensors import safe_open
from safetensors.torch import load_file, save_file


souce_path = sys.argv[1]
target_path = sys.argv[2]
# model_path = '/mnt/nas-data-1/wuchangjie.wcj/work/bev_ex3/Qwen2.5-VL-3B-Instruct'
# 加载两个部分模型
model_part1 = load_file(f"{souce_path}/model-00001-of-00002.safetensors")
model_part2 = load_file(f"{souce_path}/model-00002-of-00002.safetensors")
with safe_open(f"{souce_path}/model-00001-of-00002.safetensors", framework="pt") as f:
    metadata = f.metadata()
# 合并模型
merged_model = {**model_part1, **model_part2}
keys = list(merged_model.keys())
for key in keys:
    if "dino" in key or "vis_head" in key:
        print(f'remove key: {key}')
        merged_model.pop(key)

# 保存合并后的模型
save_file(merged_model, f"{target_path}/model.safetensors", metadata)

print("模型合并完成，保存为 model.safetensors")