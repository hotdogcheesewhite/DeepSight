# import xml.etree.ElementTree as ET
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import math
# from matplotlib.ticker import MultipleLocator

# from scipy.interpolate import CubicSpline, interp1d, splprep, splev
# import numpy as np

# def plot_smooth_route(ax, waypoints, color='blue', method='cubic_spline'):
#     """使用平滑曲线绘制路线"""
#     if len(waypoints) < 2:
#         return
    
#     # 提取x, y坐标
#     x_coords = np.array([p[0] for p in waypoints])
#     y_coords = np.array([p[1] for p in waypoints])
    
#     # 移除重复点
#     valid_indices = []
#     last_point = None
#     for i, (x, y) in enumerate(zip(x_coords, y_coords)):
#         current_point = (x, y)
#         if last_point is None or current_point != last_point:
#             valid_indices.append(i)
#             last_point = current_point
    
#     if len(valid_indices) < 2:  # 如果过滤后点数太少，就直接绘制原始点
#         ax.plot(x_coords, y_coords, '-', linewidth=1.5, color=color)
#         ax.scatter(x_coords, y_coords, color='red', s=10, alpha=0.7)
#         return
    
#     # 使用有效点
#     x_coords = x_coords[valid_indices]
#     y_coords = y_coords[valid_indices]
    
#     # 创建参数化变量t (路径长度的累积)
#     t = np.zeros(len(x_coords))
#     for i in range(1, len(x_coords)):
#         dx = x_coords[i] - x_coords[i-1]
#         dy = y_coords[i] - y_coords[i-1]
#         t[i] = t[i-1] + np.sqrt(dx*dx + dy*dy)
    
#     # 检查t是否严格递增
#     if len(np.unique(t)) != len(t):
#         # 如果有重复值，添加微小扰动使其严格递增
#         epsilon = 1e-10
#         for i in range(1, len(t)):
#             if t[i] <= t[i-1]:
#                 t[i] = t[i-1] + epsilon
    
#     # 归一化t到[0,1]区间
#     if t[-1] > 0:
#         t = t / t[-1]
    
#     try:
#         # 生成更多点用于平滑曲线
#         t_new = np.linspace(0, 1, 100)
        
#         if method == 'cubic_spline':
#             # 三次样条插值
#             cs_x = CubicSpline(t, x_coords)
#             cs_y = CubicSpline(t, y_coords)
#             x_new = cs_x(t_new)
#             y_new = cs_y(t_new)
        
#         elif method == 'b_spline':
#             # B样条插值
#             tck, u = splprep([x_coords, y_coords], s=0)
#             x_new, y_new = splev(t_new, tck)
        
#         elif method == 'pchip':
#             # PCHIP插值 (保形插值，避免过冲)
#             pchip_x = interp1d(t, x_coords, kind='cubic', assume_sorted=True)
#             pchip_y = interp1d(t, y_coords, kind='cubic', assume_sorted=True)
#             x_new = pchip_x(t_new)
#             y_new = pchip_y(t_new)
        
#         # 绘制平滑曲线
#         ax.plot(x_new, y_new, '-', linewidth=2.0, color=color)
        
#     except Exception as e:
#         print(f"平滑曲线生成失败: {e}，使用原始折线代替")
#         # 如果插值失败，回退到原始折线
#         ax.plot(x_coords, y_coords, '-', linewidth=1.5, color=color)
    
#     # 绘制原始路点
#     ax.scatter(x_coords, y_coords, color='red', s=10, alpha=0.7)

# def calculate_average_distance(points):
#     if len(points) < 2:
#         return 0
    
#     distances = []
#     for i in range(len(points) - 1):
#         point1 = np.array(points[i])
#         point2 = np.array(points[i + 1])
#         distance = np.linalg.norm(point2 - point1)  # L2距离
#         distances.append(distance)
    
#     return np.mean(distances), np.std(distances), min(distances), max(distances)

# def parse_xml_waypoints(xml_file):
#     """解析XML文件并提取所有路线的waypoints"""
#     # 保持原有代码不变
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
    
#     routes = []
    
#     for route in root.findall('route'):
#         route_id = route.get('id')
#         town = route.get('town')
        
#         waypoints_elem = route.find('waypoints')
#         if waypoints_elem is None:
#             continue
        
#         waypoints = []
#         for pos in waypoints_elem.findall('position'):
#             x = float(pos.get('x'))
#             y = float(pos.get('y'))
#             waypoints.append((x, y))
        
#         routes.append({
#             'id': route_id,
#             'town': town,
#             'waypoints': waypoints
#         })
    
#     return routes

# def set_axes_with_fixed_ticks(ax, x_min, x_max, y_min, y_max, tick_interval=10):
#     """设置坐标轴范围和统一的刻度间隔"""
#     # 向下/向上取整到10的倍数以设置合理的边界
#     x_min_adj = math.floor(x_min / tick_interval) * tick_interval
#     x_max_adj = math.ceil(x_max / tick_interval) * tick_interval
#     y_min_adj = math.floor(y_min / tick_interval) * tick_interval
#     y_max_adj = math.ceil(y_max / tick_interval) * tick_interval
    
#     # 设置坐标轴范围
#     ax.set_xlim(x_min_adj, x_max_adj)
#     ax.set_ylim(y_min_adj, y_max_adj)
    
#     # 设置主刻度为固定间隔
#     ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
#     ax.yaxis.set_major_locator(MultipleLocator(tick_interval))

# def save_waypoints_visualization(routes, output_dir="./waypoints_viz", batch_size=12, max_cols=3, figsize=(15, 15), tick_interval=10):
#     """将路线可视化成网格布局并包含密度信息，使用统一的刻度间隔"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 设置字体大小
#     plt.rcParams.update({'font.size': 10})
    
#     # 计算需要的批次数量
#     total_routes = len(routes)
#     num_batches = math.ceil(total_routes / batch_size)
    
#     print(f"共有 {total_routes} 条路线, 将分 {num_batches} 批次可视化")
    
#     # 处理每一批次
#     for batch_idx in range(num_batches):
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, total_routes)
#         batch_routes = routes[start_idx:end_idx]
        
#         # 计算网格布局
#         routes_in_batch = len(batch_routes)
#         cols = min(max_cols, routes_in_batch)
#         rows = math.ceil(routes_in_batch / cols)
        
#         # 创建图形
#         fig = plt.figure(figsize=figsize)
        
#         # 为每条路线创建子图
#         for i, route in enumerate(batch_routes):
#             route_id = route['id']
#             town = route['town']
#             waypoints = route['waypoints']
            
#             if not waypoints:
#                 print(f"警告: 路线 {route_id} 没有waypoints，跳过。")
#                 continue
            
#             # 提取x, y坐标
#             x_coords = [p[0] for p in waypoints]
#             y_coords = [p[1] for p in waypoints]
            
#             # 创建子图
#             ax = fig.add_subplot(rows, cols, i + 1)
            
#             # 绘制路线
#             # ax.plot(x_coords, y_coords, '-', linewidth=1.5, color='blue')
#             # ax.scatter(x_coords, y_coords, color='red', s=10, alpha=0.7)
#             plot_smooth_route(ax, waypoints, color='blue', method='cubic_spline')

            
#             # 标记起点和终点
#             ax.plot(x_coords[0], y_coords[0], 'go', markersize=6, label='Start')
#             ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=6, label='End')
            
#             # 计算密度信息
#             if len(x_coords) > 1 and len(y_coords) > 1:
#                 x_range = max(x_coords) - min(x_coords)
#                 y_range = max(y_coords) - min(y_coords)
#                 area = x_range * y_range
#                 density = len(waypoints) / area if area > 0 else 0
                
#                 # 添加信息文本框
#                 # info_text = f"Route ID: {route_id}\nTown: {town}\n" \
#                 #             f"Points: {len(waypoints)}\nArea: {area:.1f}\n" \
#                 #             f"Density: {density:.6f}"
#                 avg_dist, std_dist, min_dist, max_dist = calculate_average_distance(waypoints)
            
#             # 显示统计信息
#                 info_text = (
#                     f'Points: {len(waypoints)}\n'
#                     f'Area: {area:.1f}\n'
#                     f'Density: {density:.6f}\n'
#                     f'Avg dist: {avg_dist:.2f}\n'
#                     f'Std dist: {std_dist:.2f}\n'
#                     f'Min dist: {min_dist:.2f}\n'
#                     f'Max dist: {max_dist:.2f}'
#                 )
#                 ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
#                        verticalalignment='top', fontsize=8, 
#                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
#             ax.set_title(f'Route {route_id}', fontsize=10)
#             ax.set_xlabel('X', fontsize=8)
#             ax.set_ylabel('Y', fontsize=8)
#             ax.grid(True, linestyle='--', alpha=0.5)
            
#             # 使用自定义函数设置统一的刻度间隔
#             set_axes_with_fixed_ticks(ax, min(x_coords), max(x_coords), 
#                                      min(y_coords), max(y_coords), 
#                                      tick_interval=tick_interval)
        
#         # 添加整体标题
#         plt.suptitle(f'Routes Visualization (Batch {batch_idx+1}/{num_batches})', fontsize=16)
#         plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题腾出空间
        
#         # 保存图形到文件
#         output_path = os.path.join(output_dir, f"routes_batch_{batch_idx+1}_of_{num_batches}.png")
#         plt.savefig(output_path, dpi=150)
#         plt.close(fig)  # 关闭图形以释放内存
        
#         print(f"批次 {batch_idx+1}/{num_batches} 已保存到: {output_path}")
    
#     # 创建一个汇总图，显示所有路线
#     fig_all = plt.figure(figsize=(15, 10))
#     ax_all = fig_all.add_subplot(111)
#     ax_all.set_title('All Routes - Overview', fontsize=16)
#     ax_all.set_xlabel('X Coordinate', fontsize=14)
#     ax_all.set_ylabel('Y Coordinate', fontsize=14)
#     ax_all.grid(True, linestyle='--', alpha=0.7)
    
#     # 用于记录所有点的范围，以便为总览图设置适当的边界
#     all_x = []
#     all_y = []
    
#     # 使用不同的颜色表示不同路线
#     colors = plt.cm.tab20(np.linspace(0, 1, len(routes)))
    
#     for i, route in enumerate(routes):
#         waypoints = route['waypoints']
        
#         if not waypoints:
#             continue
        
#         # 提取x, y坐标
#         x_coords = [p[0] for p in waypoints]
#         y_coords = [p[1] for p in waypoints]
        
#         all_x.extend(x_coords)
#         all_y.extend(y_coords)
        
#         color = colors[i % len(colors)]  # 循环使用颜色
#         # ax_all.plot(x_coords, y_coords, '-', linewidth=1.0, color=color, alpha=0.7)
#         # 使用平滑曲线绘制
#         plot_smooth_route(ax_all, waypoints, color=color, method='cubic_spline')
    
#     # 为汇总图设置统一的刻度间隔
#     if all_x and all_y:
#         set_axes_with_fixed_ticks(ax_all, min(all_x), max(all_x), min(all_y), max(all_y), tick_interval=tick_interval)
    
#     # 添加汇总信息
#     ax_all.text(0.02, 0.98, f"Total Routes: {len(routes)}", transform=ax_all.transAxes, 
#                verticalalignment='top', bbox=dict(boxstyle='round', 
#                facecolor='wheat', alpha=0.5))
    
#     # 保存汇总图
#     plt.tight_layout()
#     fig_all.savefig(os.path.join(output_dir, "all_routes_overview.png"), dpi=200)
#     plt.close(fig_all)
    
#     print(f"已将全部可视化结果保存到目录: {output_dir}")
#     print(f"共处理了 {len(routes)} 条路线")

# def main():
#     xml_file = "/home/zhanglingjun.zlj/code/Bench2Drive/leaderboard/data/bench2drive220.xml"
    
#     if not os.path.exists(xml_file):
#         print(f"错误: 找不到文件 {xml_file}")
#         return
    
#     print(f"正在解析文件: {xml_file}")
#     routes = parse_xml_waypoints(xml_file)
    
#     if not routes:
#         print("警告: 未找到任何路线数据")
#         return
    
#     print(f"找到 {len(routes)} 条路线")
#     # 可以在这里调整刻度间隔，默认为10
#     save_waypoints_visualization(routes, output_dir="./visualization_results_2202", batch_size=12, max_cols=3, tick_interval=20)

# if __name__ == "__main__":
#     main()
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from matplotlib.ticker import MultipleLocator
import numpy as np
def sample_path_equidistant(waypoints, interval=5.0):
    """
    从起点开始沿路径进行等距离采样
    
    参数:
    - waypoints: 路径点列表 [(x1,y1), (x2,y2), ...]
    - interval: 采样间隔（L2距离单位）
    
    返回:
    - 采样点列表 [(x1,y1), (x2,y2), ...]
    """
    if len(waypoints) < 2:
        return waypoints  # 不足两点直接返回
    
    samples = [waypoints[0]]  # 起点
    current_dist = 0.0
    next_sample_dist = interval
    
    for i in range(1, len(waypoints)):
        p0 = np.array(waypoints[i-1])
        p1 = np.array(waypoints[i])
        seg_len = np.linalg.norm(p1 - p0)
        
        # 在当前线段上生成所有采样点
        while next_sample_dist <= current_dist + seg_len:
            ratio = (next_sample_dist - current_dist) / seg_len
            sample_point = p0 + ratio * (p1 - p0)
            samples.append(tuple(sample_point))
            next_sample_dist += interval
        
        current_dist += seg_len
    
    return samples
def calculate_angle(v1, v2):
    """计算两个向量之间的夹角（度）"""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0
    
    cos_theta = dot / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def line_intersection(line1, line2):
    """计算两条直线的交点"""
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if abs(div) < 1e-8:
        return None  # 平行线

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)
def crop_sharp_turns_improved(waypoints, angle_threshold=5, max_extension=150.0):
    """
    使用前后两个点定义的直线交点裁剪急转弯点，并避免重复处理参与计算的点
    
    参数:
    - waypoints: 原始轨迹点列表
    - angle_threshold: 角度阈值（度），超过此值视为急转弯
    - max_extension: 裁剪点与原点的最大允许距离
    
    返回:
    - 裁剪后的轨迹点列表
    """
    if len(waypoints) < 5:  # 至少需要5个点才能获取前后各两个点
        return waypoints
    
    points = np.array(waypoints, dtype=float)
    cropped = [tuple(points[0])]  # 保留起点
    
    # 处理第二个点
    cropped.append(tuple(points[1]))
    
    i = 2  # 从第三个点开始处理
    while i < len(points) - 2:  # 直到倒数第三个点结束
        # 计算前后向量
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        
        # 计算夹角
        angle = calculate_angle(v1, v2)
        
        if angle > angle_threshold:
            # 定义前面的直线：使用 (i-2, i-1) 这两个点
            line1 = (tuple(points[i-2]), tuple(points[i-1]))
            
            # 定义后面的直线：使用 (i+1, i+2) 这两个点
            line2 = (tuple(points[i+1]), tuple(points[i+2]))
            
            # 计算两条直线的交点
            intersection = line_intersection(line1, line2)
            
            if intersection:
                # 检查交点是否合理（距离不能太远）
                dist_to_turn = np.linalg.norm(np.array(intersection) - points[i])
                if dist_to_turn < max_extension:
                    # 用交点替代当前转弯点
                    cropped.append(intersection)
                    
                    # 关键改进：跳过被处理过的点
                    # 跳过当前点和后面两个参与计算的点
                    i += 3  # 直接跳到 i+3
                    continue
        
        # 没有急转弯或交点无效，保留当前点
        cropped.append(tuple(points[i]))
        i += 1  # 正常递增
    
    # 添加剩余未处理的点
    for j in range(max(i, len(points)-2), len(points)):
        cropped.append(tuple(points[j]))
    
    # 移除重复点
    unique_points = []
    for p in cropped:
        if not unique_points or np.linalg.norm(np.array(p) - np.array(unique_points[-1])) > 1e-5:
            unique_points.append(p)
    
    return unique_points

def plot_sampled_points(ax, waypoints, color='blue', angle_threshold=15, max_extension=5000.0):
    """
    仅绘制等距离采样点（L2距离=5）
    
    参数:
    - ax: matplotlib轴对象
    - waypoints: 原始轨迹点列表
    - color: 采样点颜色
    - angle_threshold: 角度阈值（度）
    - max_extension: 裁剪点与原点的最大允许距离
    
    返回:
    - 绘制的散点对象
    """
    if len(waypoints) < 2:
        return None
    
    # 1. 移除重复点
    filtered = []
    last = None
    for p in waypoints:
        if last is None or np.linalg.norm(np.array(p) - np.array(last)) > 1e-5:
            filtered.append(p)
            last = p
    
    if len(filtered) < 2:
        return None
    
    # 2. 应用急转弯裁剪
    cropped = crop_sharp_turns_improved(filtered, angle_threshold, max_extension)
    
    # 3. 等距离采样 (L2距离=5)
    samples = sample_path_equidistant(cropped, interval=10)
    
    # 4. 仅绘制采样点
    if samples:
        x = [p[0] for p in samples]
        y = [p[1] for p in samples]
        return ax.scatter(x, y, color=color, s=15, alpha=0.8, zorder=5)
    
    return None
def plot_cropped_route(ax, waypoints, color='blue'):
    """使用射线交点裁剪方法绘制路线"""
    if len(waypoints) < 2:
        return
    
    # 裁剪急转弯
    cropped_waypoints = crop_sharp_turns_improved(waypoints)
    
    # 提取坐标
    x_coords = [p[0] for p in cropped_waypoints]
    y_coords = [p[1] for p in cropped_waypoints]
    
    # 绘制裁剪后的轨迹
    ax.plot(x_coords, y_coords, '-', linewidth=2.0, color=color)
    
    # 绘制裁剪后的路点
    ax.scatter(x_coords, y_coords, color='red', s=10, alpha=0.7)
    
    # 标记交点（绿色）
    for i in range(1, len(cropped_waypoints)-1):
        if i < len(waypoints) - 1 and np.linalg.norm(
            np.array(cropped_waypoints[i]) - np.array(waypoints[i])
        ) > 1e-3:
            ax.plot(cropped_waypoints[i][0], cropped_waypoints[i][1], 
                   'go', markersize=6, alpha=0.7)

def calculate_average_distance(points):
    if len(points) < 2:
        return 0
    
    distances = []
    for i in range(len(points) - 1):
        point1 = np.array(points[i])
        point2 = np.array(points[i + 1])
        distance = np.linalg.norm(point2 - point1)  # L2距离
        distances.append(distance)
    
    return np.mean(distances), np.std(distances), min(distances), max(distances)

def parse_xml_waypoints(xml_file):
    """解析XML文件并提取所有路线的waypoints"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    routes = []
    
    for route in root.findall('route'):
        route_id = route.get('id')
        town = route.get('town')
        
        waypoints_elem = route.find('waypoints')
        if waypoints_elem is None:
            continue
        
        waypoints = []
        for pos in waypoints_elem.findall('position'):
            x = float(pos.get('x'))
            y = float(pos.get('y'))
            waypoints.append((x, y))
        
        routes.append({
            'id': route_id,
            'town': town,
            'waypoints': waypoints
        })
    
    return routes

def set_axes_with_fixed_ticks(ax, x_min, x_max, y_min, y_max, tick_interval=10):
    """设置坐标轴范围和统一的刻度间隔"""
    x_min_adj = math.floor(x_min / tick_interval) * tick_interval
    x_max_adj = math.ceil(x_max / tick_interval) * tick_interval
    y_min_adj = math.floor(y_min / tick_interval) * tick_interval
    y_max_adj = math.ceil(y_max / tick_interval) * tick_interval
    
    ax.set_xlim(x_min_adj, x_max_adj)
    ax.set_ylim(y_min_adj, y_max_adj)
    
    ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
    ax.yaxis.set_major_locator(MultipleLocator(tick_interval))

def save_waypoints_visualization(routes, output_dir="./waypoints_viz", batch_size=12, max_cols=3, figsize=(15, 15), tick_interval=10):
    """将路线可视化成网格布局并包含密度信息，使用统一的刻度间隔"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams.update({'font.size': 10})
    
    total_routes = len(routes)
    num_batches = math.ceil(total_routes / batch_size)
    
    print(f"共有 {total_routes} 条路线, 将分 {num_batches} 批次可视化")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_routes)
        batch_routes = routes[start_idx:end_idx]
        
        routes_in_batch = len(batch_routes)
        cols = min(max_cols, routes_in_batch)
        rows = math.ceil(routes_in_batch / cols)
        
        fig = plt.figure(figsize=figsize)
        
        for i, route in enumerate(batch_routes):
            route_id = route['id']
            town = route['town']
            waypoints = route['waypoints']
            
            if not waypoints:
                print(f"警告: 路线 {route_id} 没有waypoints，跳过。")
                continue
            
            x_coords = [p[0] for p in waypoints]
            y_coords = [p[1] for p in waypoints]
            
            ax = fig.add_subplot(rows, cols, i + 1)
            
            # 使用射线交点裁剪方法绘制路线
            plot_sampled_points(ax, waypoints, color='blue')
            
            # 标记起点和终点
            ax.plot(x_coords[0], y_coords[0], 'go', markersize=6, label='Start')
            ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=6, label='End')
            
            # 计算密度信息
            if len(x_coords) > 1 and len(y_coords) > 1:
                x_range = max(x_coords) - min(x_coords)
                y_range = max(y_coords) - min(y_coords)
                area = x_range * y_range
                density = len(waypoints) / area if area > 0 else 0
                
                avg_dist, std_dist, min_dist, max_dist = calculate_average_distance(waypoints)
            
                info_text = (
                    f'Points: {len(waypoints)}\n'
                    f'Area: {area:.1f}\n'
                    f'Density: {density:.6f}\n'
                    f'Avg dist: {avg_dist:.2f}\n'
                    f'Std dist: {std_dist:.2f}\n'
                    f'Min dist: {min_dist:.2f}\n'
                    f'Max dist: {max_dist:.2f}'
                )
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(f'Route {route_id}', fontsize=10)
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            set_axes_with_fixed_ticks(ax, min(x_coords), max(x_coords), 
                                     min(y_coords), max(y_coords), 
                                     tick_interval=tick_interval)
        
        plt.suptitle(f'Routes Visualization (Batch {batch_idx+1}/{num_batches})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        output_path = os.path.join(output_dir, f"routes_batch_{batch_idx+1}_of_{num_batches}.png")
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        
        print(f"批次 {batch_idx+1}/{num_batches} 已保存到: {output_path}")
    
    # 创建汇总图
    fig_all = plt.figure(figsize=(15, 10))
    ax_all = fig_all.add_subplot(111)
    ax_all.set_title('All Routes - Overview', fontsize=16)
    ax_all.set_xlabel('X Coordinate', fontsize=14)
    ax_all.set_ylabel('Y Coordinate', fontsize=14)
    ax_all.grid(True, linestyle='--', alpha=0.7)
    
    all_x = []
    all_y = []
    colors = plt.cm.tab20(np.linspace(0, 1, len(routes)))
    
    for i, route in enumerate(routes):
        waypoints = route['waypoints']
        
        if not waypoints:
            continue
        
        x_coords = [p[0] for p in waypoints]
        y_coords = [p[1] for p in waypoints]
        
        all_x.extend(x_coords)
        all_y.extend(y_coords)
        
        color = colors[i % len(colors)]
        plot_sampled_points(ax_all, waypoints, color=color)
    
    if all_x and all_y:
        set_axes_with_fixed_ticks(ax_all, min(all_x), max(all_x), min(all_y), max(all_y), tick_interval=tick_interval)
    
    ax_all.text(0.02, 0.98, f"Total Routes: {len(routes)}", transform=ax_all.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig_all.savefig(os.path.join(output_dir, "all_routes_overview.png"), dpi=200)
    plt.close(fig_all)
    
    print(f"已将全部可视化结果保存到目录: {output_dir}")
    print(f"共处理了 {len(routes)} 条路线")

def main():
    xml_file = "/home/zhanglingjun.zlj/code/Bench2Drive/leaderboard/data/bench2drive220.xml"
    
    if not os.path.exists(xml_file):
        print(f"错误: 找不到文件 {xml_file}")
        return
    
    print(f"正在解析文件: {xml_file}")
    routes = parse_xml_waypoints(xml_file)
    
    if not routes:
        print("警告: 未找到任何路线数据")
        return
    
    print(f"找到 {len(routes)} 条路线")
    save_waypoints_visualization(routes, output_dir="./visualization_results_2204", batch_size=12, max_cols=3, tick_interval=20)

if __name__ == "__main__":
    main()
