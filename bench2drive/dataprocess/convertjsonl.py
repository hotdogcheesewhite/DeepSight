import ijson
import json
from tqdm import tqdm

INPUT_JSON = "/home/zhanglingjun.zlj/code/road2rome/gt_indices_train_full.json"
OUTPUT_JSONL = "/home/zhanglingjun.zlj/code/road2rome/gt_indices_train_fulljsonl.jsonl"

# 先快速统计总 key 数量（可选，用于进度条 total）
print("Counting total keys... (this may take a moment)")
total_keys = 0
with open(INPUT_JSON, 'rb') as f:
    for _ in ijson.kvitems(f, ''):
        total_keys += 1

print(f"Total keys to process: {total_keys}")

# 开始转换
with open(INPUT_JSON, 'rb') as fin, open(OUTPUT_JSONL, 'w', encoding='utf-8') as fout:
    items = ijson.kvitems(fin, '')  # 流式遍历根对象的 key-value
    with tqdm(total=total_keys, desc="Converting", unit="keys") as pbar:
        for k, v in items:
            if isinstance(v, dict) and 'CAM_FRONT' in v:
                result_str = v['CAM_FRONT']
            else:
                result_str = v
            line = json.dumps({"id": str(k), "result": result_str}, ensure_ascii=False)
            fout.write(line + '\n')
            pbar.update(1)

print("✅ Conversion completed!")