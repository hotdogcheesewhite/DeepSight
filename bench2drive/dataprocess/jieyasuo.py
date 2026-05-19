# import numpy as np
# import os
# import time
# import argparse

# def extract_npz(npz_file, output_dir):
#     """将 NPZ 文件解压到指定目录"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     print(f"开始解压文件: {npz_file} 到 {output_dir}")
#     start_time = time.time()
    
#     try:
#         data = np.load(npz_file, allow_pickle=True)
        
#         for i, name in enumerate(data.files):
#             array = data[name]
#             output_path = os.path.join(output_dir, f"{name}.npy")
#             np.save(output_path, array)
#             print(f"({i+1}/{len(data.files)}) 已保存: '{name}' - 形状: {array.shape}")
            
#         elapsed_time = time.time() - start_time
#         print(f"解压完成. 共解压 {len(data.files)} 个数组. 用时: {elapsed_time:.2f} 秒")
        
#     except Exception as e:
#         print(f"解压过程中出错: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='解压 NPZ 文件到指定目录')
#     parser.add_argument('npz_file', nargs='?', 
#                         default="/mnt/nas-data-1/zhanglingjun.zlj1/data/b2dmap/Town02_HD_map.npz", 
#                         help='NPZ 文件路径')
#     parser.add_argument('output_dir', nargs='?',
#                         default="/home/zhanglingjun.zlj/data/b2dmap/Town02_HD_map",
#                         help='输出目录路径')
    
#     args = parser.parse_args()
#     extract_npz(args.npz_file, args.output_dir)
import numpy as np
import json

# 1. 读取 .npy 文件
array = np.load('/home/zhanglingjun.zlj/code/Bench2Drive/arr.npy',allow_pickle=True)

# 2. 将 NumPy 数组转换为 Python 原生类型
def numpy_to_python(obj):
    """递归转换 NumPy 对象为 Python 原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    return obj

# 3. 转换数据并保存为 JSON
python_data = numpy_to_python(array)

# 4. 保存为 JSON 文件
with open('/home/zhanglingjun.zlj/code/Bench2Drive/town2.json', 'w') as f:
    json.dump(python_data, f, indent=2)

print("转换完成! 已保存到: /home/zhanglingjun.zlj/code/Bench2Drive/arr.json")
