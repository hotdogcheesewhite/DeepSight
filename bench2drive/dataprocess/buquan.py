import json

def extract_dicts_with_key_6(json_file_path, output_file_path):
    """
    提取所有第一个元素为6的内部列表的第二个元素（字典）
    """
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 查找所有第一个元素是6的项
    extracted_data = []
    for item in data:
        # 检查是否是列表，长度至少为2，且第一个元素是6
        if isinstance(item, list) and len(item) >= 2 and item[0] == 11:
            extracted_data.append(item[1])
    
    # 保存提取结果
    with open(output_file_path, 'w') as f:
        json.dump(extracted_data, f, indent=2)
    
    return extracted_data

# 使用脚本
json_path = '/home/zhanglingjun.zlj/code/Bench2Drive/town2.json'
output_path = '/home/zhanglingjun.zlj/code/Bench2Drive/extracted_6.json'

results = extract_dicts_with_key_6(json_path, output_path)
print(f"成功提取 {len(results)} 个元素，已保存到: {output_path}")
