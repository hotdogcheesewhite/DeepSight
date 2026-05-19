import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

def extract_and_visualize_routes(json_file_path, output_dir='./routes'):
    """
    提取JSON中的坐标点并可视化路线
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 用于存储不同路线的坐标
    routes = {}
    
    # 遍历所有数据项
    for item in data:
        # 遍历每个键（如"-1", "1"等）
        for key, value in item.items():
            if isinstance(value, list):
                # 初始化该路线的坐标列表
                if key not in routes:
                    routes[key] = []
                
                # 遍历列表中的每个元素
                for point_group in value:
                    if "Points" in point_group:
                        for point_data in point_group["Points"]:
                            # 提取坐标点 [x, y, z]
                            coord = point_data[0]
                            routes[key].append(coord)
    
    # 可视化每条路线
    for key, coords in routes.items():
        if not coords:
            continue
        
        # 转换为NumPy数组以便处理
        coords = np.array(coords)
        
        # 提取x和y坐标（忽略z坐标，因为我们只关心2D平面）
        x = coords[:, 0]
        y = coords[:, 1]
        
        # 创建颜色映射（根据路线长度渐变）
        colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        plt.title(f'Route {key}', fontsize=16)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制路线，使用颜色渐变表示前进方向
        for i in range(len(x) - 1):
            plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color=colors[i], linewidth=2.5)
        
        # 添加起点和终点标记
        plt.scatter(x[0], y[0], color='green', s=100, zorder=5, label='Start')
        plt.scatter(x[-1], y[-1], color='red', s=100, zorder=5, label='End')
        
        # 添加图例
        plt.legend(loc='best')
        
        # 保持纵横比
        plt.axis('equal')
        
        # 保存图像
        output_path = os.path.join(output_dir, f'route_{key}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成路线图: {output_path} (包含 {len(coords)} 个点)")
    
    return routes

if __name__ == "__main__":
    # 配置路径
    json_path = '/home/zhanglingjun.zlj/code/Bench2Drive/extracted_6.json'
    output_dir = '/home/zhanglingjun.zlj/code/Bench2Drive/testroutes'
    
    # 执行提取和可视化
    routes = extract_and_visualize_routes(json_path, output_dir)
    
    # 打印统计信息
    print("\n路线统计:")
    for key, coords in routes.items():
        print(f"  - 路线 {key}: {len(coords)} 个点")
