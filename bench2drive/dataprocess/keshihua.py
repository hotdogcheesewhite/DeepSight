import json
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from scipy.interpolate import CubicSpline, interp1d, splprep, splev

def sample_path_equidistant(waypoints, sample_interval=1.5):
    """
    从起点开始沿路径进行等距离采样
    
    参数:
    - waypoints: 路径点列表 [(x1,y1), (x2,y2), ...]
    - sample_interval: 采样间隔（L2距离单位）
    
    返回:
    - 采样点列表 [(x1,y1), (x2,y2), ...]
    """
    if len(waypoints) < 2:
        return waypoints  # 不足两点无法采样
    
    # 计算路径总长度
    total_length = 0.0
    segment_lengths = []
    
    for i in range(1, len(waypoints)):
        p1 = np.array(waypoints[i-1])
        p2 = np.array(waypoints[i])
        seg_len = np.linalg.norm(p2 - p1)
        segment_lengths.append(seg_len)
        total_length += seg_len
    
    # 生成采样点
    samples = []
    current_distance = 0.0
    next_sample_distance = sample_interval
    
    # 添加起点
    samples.append(waypoints[0])
    
    # 遍历每个线段
    for i in range(len(segment_lengths)):
        seg_start = current_distance
        seg_end = current_distance + segment_lengths[i]
        
        # 在当前线段内生成所有采样点
        while next_sample_distance < seg_end:
            # 计算线段上的比例位置
            ratio = (next_sample_distance - seg_start) / segment_lengths[i]
            p0 = np.array(waypoints[i])
            p1 = np.array(waypoints[i+1])
            sample_point = p0 + ratio * (p1 - p0)
            
            samples.append(tuple(sample_point))
            next_sample_distance += sample_interval
        
        current_distance = seg_end
    
    # 添加终点（如果需要） - 但题目要求等距离采样，通常不包含终点
    # 如果要求包含终点，可取消下面注释
    if samples[-1] != waypoints[-1]:
        samples.append(waypoints[-1])
    
    return samples

def plot_sampled_points(ax, waypoints, color='blue', angle_threshold=15, 
                        max_extension=5000.0, sample_interval=10, label=None):
    """
    绘制等距离采样点（不绘制路径线）
    
    参数:
    - ax: matplotlib轴对象
    - waypoints: 原始轨迹点列表
    - color: 点的颜色
    - angle_threshold: 角度阈值（度）
    - max_extension: 裁剪点与原点的最大允许距离
    - sample_interval: 采样间隔（L2距离单位）
    - label: 图例标签
    
    返回:
    - 绘制的散点对象
    """
    if len(waypoints) < 2:
        return None
    
    # 1. 移除重复点
    valid_indices = []
    last_point = None
    for i, point in enumerate(waypoints):
        current_point = np.array(point)
        if last_point is None or np.linalg.norm(current_point - last_point) > 1e-5:
            valid_indices.append(i)
            last_point = current_point
    
    if len(valid_indices) < 2:  # 至少需要2个点才能形成路径
        return None
    
    filtered_waypoints = [waypoints[i] for i in valid_indices]
    
    # 2. 应用急转弯裁剪
    cropped_waypoints = crop_sharp_turns_improved(
        filtered_waypoints, 
        angle_threshold, 
        max_extension
    )
    
    # 3. 等距离采样
    sampled_points = sample_path_equidistant(cropped_waypoints, sample_interval)
    
    
    return sampled_points

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
def plot_cropped_route(ax, waypoints, color='blue', angle_threshold=15, max_extension=5000.0, label=None):
    """
    使用射线交点裁剪方法优化并绘制路线，直接使用折线连接
    
    参数:
    - ax: matplotlib轴对象
    - waypoints: 原始轨迹点列表
    - color: 绘图颜色
    - angle_threshold: 角度阈值（度），超过此值视为急转弯
    - max_extension: 裁剪点与原点的最大允许距离
    - label: 图例标签
    
    返回:
    - 绘制的线对象
    """
    if len(waypoints) < 2:
        return None
    
    # 提取x, y坐标
    x_coords = np.array([p[0] for p in waypoints])
    y_coords = np.array([p[1] for p in waypoints])
    
    # 移除重复点
    valid_indices = []
    last_point = None
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        current_point = (x, y)
        if last_point is None or np.linalg.norm(np.array(current_point) - np.array(last_point)) > 1e-5:
            valid_indices.append(i)
            last_point = current_point
    
    if len(valid_indices) < 5:  # 如果过滤后点数太少，就直接绘制原始点
        line = ax.plot(x_coords, y_coords, '-', linewidth=1.5, color=color, label=label)[0]
        ax.scatter(x_coords, y_coords, color='red', s=10, alpha=0.7)
        return line
    
    # 使用有效点
    x_coords = x_coords[valid_indices]
    y_coords = y_coords[valid_indices]
    filtered_waypoints = [(x, y) for x, y in zip(x_coords, y_coords)]
    
    # 应用急转弯裁剪算法
    cropped_waypoints = crop_sharp_turns_improved(filtered_waypoints, angle_threshold, max_extension)
    
    # 提取裁剪后的坐标
    cropped_x = [p[0] for p in cropped_waypoints]
    cropped_y = [p[1] for p in cropped_waypoints]
    
    # 绘制裁剪后的路径（不使用曲线拟合）
    line = ax.plot(cropped_x, cropped_y, '-', linewidth=2.0, color=color, label=label)[0]
    
    # 绘制裁剪后的路点
    ax.scatter(cropped_x, cropped_y, color='red', s=10, alpha=0.7)
    
    # 标记原始急转弯点（可选）
    # for i in range(1, len(filtered_waypoints)-1):
    #     v1 = np.array(filtered_waypoints[i]) - np.array(filtered_waypoints[i-1])
    #     v2 = np.array(filtered_waypoints[i+1]) - np.array(filtered_waypoints[i])
    #     angle = calculate_angle(v1, v2)
    #     if angle > angle_threshold:
    #         ax.plot(filtered_waypoints[i][0], filtered_waypoints[i][1], 
    #                'go', markersize=6, alpha=0.5)
    
    return line

def crop_sharp_turns_improved(waypoints, angle_threshold=45, max_extension=50.0):
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
    angle_threshold_rad = np.deg2rad(angle_threshold)

    while i < len(points) - 2:  # 直到倒数第三个点结束
        # 计算前后向量
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        
        # 计算夹角
        angle = calculate_angle(v1, v2)
        
        if angle > angle_threshold_rad:
            # 定义前面的直线：使用 (i-2, i-1) 这两个点
            line1 = (tuple(points[i-2]), tuple(points[i-1]))
            
            # 定义后面的直线：使用 (i+1, i+2) 这两个点
            line2 = (tuple(points[i+1]), tuple(points[i+2]))
            
            # 计算两条直线的交点
            intersection = line_intersection(line1, line2)
            
            if intersection:
                p = np.array(intersection)
                # 检查条件1：交点在第一条直线的正向延长线 (i-1 -> 方向)
                # 向量 AB = points[i-1] - points[i-2] (线段方向)
                # 向量 BP = p - points[i-1] (从i-1到交点)
                AB = points[i-1] - points[i-2]
                BP = p - points[i-1]
                # 要求 BP 与 AB 同向 (点乘 > 0)
                if np.dot(BP, AB) <= 0:
                    i += 1
                    continue
                CD = points[i+2] - points[i+1]
                CP = p - points[i+1]
                # 要求 CP 与 CD 反向 (点乘 < 0)
                if np.dot(CP, CD) >= 0:
                    i += 1
                    continue
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





def calculate_average_distance(points):
    if len(points) < 2:
        return 0, 0, 0, 0
    
    distances = []
    for i in range(len(points) - 1):
        point1 = np.array(points[i])
        point2 = np.array(points[i + 1])
        distance = np.linalg.norm(point2 - point1)  # L2距离
        distances.append(distance)
    
    return np.mean(distances), np.std(distances), min(distances), max(distances)

# 读取JSONL文件
def read_points_from_jsonl(file_path):
    points_by_line = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # 解析每一行的JSON数据
                points = json.loads(line.strip())
                points_by_line.append(points)
            except json.JSONDecodeError:
                print(f"警告: 无法解析行: {line.strip()}")
    return points_by_line

# 计算两向量之间的角度（弧度）
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 避免除零错误
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    # 确保dot_product/norm_v1/norm_v2在[-1, 1]范围内
    cos_angle = max(min(dot_product / (norm_v1 * norm_v2), 1.0), -1.0)
    return math.acos(cos_angle)

def extract_straight_lanes(points, angle_threshold=0.3, min_turn_length=5, neighborhood_size=3):
    """
    仅保留转弯前后的直线车道，移除整个转弯段
    返回值与原函数保持一致: (保留点列表, 移除点索引列表)
    
    参数:
    - points: 轨迹点列表 [(x1,y1), (x2,y2), ...]
    - angle_threshold: 方向突变阈值（弧度）, 0.3≈17°
    - min_turn_length: 最小转弯段长度（防止单点噪声误判）
    - neighborhood_size: 方向计算邻域大小
    
    返回:
    - straight_points: 仅含直线车道的点列表（转弯前+转弯后）
    - removed_indices: 被移除的转弯段点的索引列表
    """
    n = len(points)
    if n < 2 * neighborhood_size + min_turn_length:
        return points, []  # 无转弯段，返回原轨迹
    
    # 步骤1: 计算每个点的方向变化曲率
    curvatures = np.zeros(n)
    for i in range(neighborhood_size, n - neighborhood_size):
        prev_vec = np.array(points[i]) - np.array(points[i - neighborhood_size])
        next_vec = np.array(points[i + neighborhood_size]) - np.array(points[i])
        curvatures[i] = calculate_angle(prev_vec, next_vec)
    
    # 步骤2: 识别连续转弯段
    turning = curvatures > angle_threshold
    turn_segments = []
    
    i = 0
    while i < n:
        if turning[i]:
            start = i
            while i < n and turning[i]:
                i += 1
            end = i - 1
            if end - start + 1 >= min_turn_length:
                turn_segments.append((start, end))
        else:
            i += 1
    
    # 步骤3: 处理无转弯情况
    if not turn_segments:
        return points, []
    
    # 步骤4: 取第一个有效转弯段（右转场景通常只有一次转弯）
    turn_start, turn_end = turn_segments[0]

    removed_distance = 0
    removed_indices = list(range(turn_start, turn_end + 1))
    removed_points = [points[i] for i in removed_indices]
    for i in range(len(removed_points) - 1):
        p1 = np.array(removed_points[i])
        p2 = np.array(removed_points[i + 1])
        segment_distance = np.linalg.norm(p2 - p1)  # 计算相邻两点的欧几里得距离
        removed_distance += segment_distance
    # 步骤5: 构建结果
    if removed_distance<=30:
        straight_points = points[:turn_start] + points[turn_end + 1:]
        removed_indices = list(range(turn_start, turn_end + 1))
    else:
        removed_indices = list(range(turn_start, turn_end + 1))
        straight_points = points
    
    return straight_points, removed_indices

# 检测并移除转弯处的点
# def remove_turn_points(points, angle_threshold=0.3, neighborhood_size=5):
#     """
#     移除转弯处的点
    
#     参数:
#     - points: 点的列表 [(x1,y1), (x2,y2), ...]
#     - angle_threshold: 判定为转弯的角度阈值（弧度）
#     - neighborhood_size: 计算方向变化时考虑的邻域大小
    
#     返回:
#     - 移除转弯点后的点列表
#     - 被移除的点的索引列表
#     """
#     if len(points) <= 2 * neighborhood_size + 1:
#         return points, []
    
#     # 计算每一点的方向变化
#     removed_indices = []
    
#     # 保留起点和终点
#     keep_indices = [0, len(points)-1]

#     sumdis = 0
    
#     for i in range(neighborhood_size, len(points) - neighborhood_size):
#         # 计算前后向量
#         prev_vector = np.array(points[i]) - np.array(points[i - neighborhood_size])
#         next_vector = np.array(points[i + neighborhood_size]) - np.array(points[i])
        
#         # 计算角度
#         angle = calculate_angle(prev_vector, next_vector)
        
#         # 如果角度大于阈值，标记为转弯点
#         if angle > angle_threshold:
#             removed_indices.append(i)
#         else:
#             keep_indices.append(i)
    
#     # 移除转弯点
#     for i in range(len(removed_indices) - 1):
#         idx1 = removed_indices[i]
#         idx2 = removed_indices[i + 1]
#         point1 = np.array(points[idx1])
#         point2 = np.array(points[idx2])
#         sumdis += np.linalg.norm(point2 - point1)
    
#     if sumdis <=20:
#         keep_indices = sorted(list(set(keep_indices)))  # 去重并排序
#         reduced_points = [points[i] for i in keep_indices]
#     else:
#         reduced_points = points
    
#     return reduced_points, removed_indices



# 平滑曲线绘制函数
def plot_smooth_route(ax, waypoints, color='blue', method='cubic_spline', label=None):
    """使用平滑曲线绘制路线"""
    if len(waypoints) < 2:
        return
    
    # 提取x, y坐标
    x_coords = np.array([p[0] for p in waypoints])
    y_coords = np.array([p[1] for p in waypoints])
    
    # 移除重复点
    valid_indices = []
    last_point = None
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        current_point = (x, y)
        if last_point is None or current_point != last_point:
            valid_indices.append(i)
            last_point = current_point
    
    if len(valid_indices) < 2:  # 如果过滤后点数太少，就直接绘制原始点
        line = ax.plot(x_coords, y_coords, '-', linewidth=1.5, color=color, label=label)[0]
        ax.scatter(x_coords, y_coords, color='red', s=10, alpha=0.7)
        return line
    
    # 使用有效点
    x_coords = x_coords[valid_indices]
    y_coords = y_coords[valid_indices]
    
    # 创建参数化变量t (路径长度的累积)
    t = np.zeros(len(x_coords))
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i-1]
        dy = y_coords[i] - y_coords[i-1]
        t[i] = t[i-1] + np.sqrt(dx*dx + dy*dy)
    
    # 检查t是否严格递增
    if len(np.unique(t)) != len(t):
        # 如果有重复值，添加微小扰动使其严格递增
        epsilon = 1e-10
        for i in range(1, len(t)):
            if t[i] <= t[i-1]:
                t[i] = t[i-1] + epsilon
    
    # 归一化t到[0,1]区间
    if t[-1] > 0:
        t = t / t[-1]
    
    try:
        # 生成更多点用于平滑曲线
        t_new = np.linspace(0, 1, 100)
        
        if method == 'cubic_spline':
            # 三次样条插值
            cs_x = CubicSpline(t, x_coords)
            cs_y = CubicSpline(t, y_coords)
            x_new = cs_x(t_new)
            y_new = cs_y(t_new)
        
        elif method == 'b_spline':
            # B样条插值
            tck, u = splprep([x_coords, y_coords], s=0)
            x_new, y_new = splev(t_new, tck)
        
        elif method == 'pchip':
            # PCHIP插值 (保形插值，避免过冲)
            pchip_x = interp1d(t, x_coords, kind='cubic', assume_sorted=True)
            pchip_y = interp1d(t, y_coords, kind='cubic', assume_sorted=True)
            x_new = pchip_x(t_new)
            y_new = pchip_y(t_new)
        
        # 绘制平滑曲线
        line = ax.plot(x_new, y_new, '-', linewidth=2.0, color=color, label=label)[0]
        
    except Exception as e:
        print(f"平滑曲线生成失败: {e}，使用原始折线代替")
        # 如果插值失败，回退到原始折线
        line = ax.plot(x_coords, y_coords, '-', linewidth=1.5, color=color, label=label)[0]
    
    # 绘制原始路点
    ax.scatter(x_coords, y_coords, color='red', s=10, alpha=0.7)
    
    return line

# 在指定Axes上绘制对比图（不再创建新Figure）
def plot_comparison_on_axis(ax, points, index):
    """
    在给定的Axes对象上绘制对比图
    
    参数:
    - ax: matplotlib的Axes对象
    - points: 一条路线的点数据
    - index: 路线在批次中的索引（用于标题）
    """
    # 移除转弯点
    reduced_points, removed_indices = extract_straight_lanes(points, angle_threshold=0.05)
    
    # 绘制原始路线
    # ax.plot([p[0] for p in points], [p[1] for p in points], '-o', 
    #         color='blue', alpha=0.5, markersize=3, linewidth=1, label='Original')
    
    # 绘制移除转弯点后的折线
    # ax.plot([p[0] for p in reduced_points], [p[1] for p in reduced_points], '-o', 
    #         color='red', alpha=0.7, markersize=4, linewidth=1.5, label='Reduced')
    
    # 使用平滑曲线方法重新连接
    plot_sampled_points(ax, reduced_points, color='blue', angle_threshold=15, max_extension=25, label='Route 1')
    # plot_cropped_route(ax, reduced_points, color='blue', angle_threshold=15, max_extension=25, label='Route 1')
    # plot_smooth_route(ax, reduced_points, color='green', method='cubic_spline', label='Smoothed')
    
    # 标记被移除的点
    removed_points = [points[i] for i in removed_indices]
    if removed_points:
        ax.scatter([p[0] for p in removed_points], [p[1] for p in removed_points], 
                   color='orange', s=30, alpha=0.7, marker='x', label='Removed')
    
    # 设置固定刻度为20
    if points:
        # 确定坐标范围
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 计算刻度位置，以20为间隔
        x_ticks = np.arange(math.floor(x_min/20)*20, math.ceil(x_max/20)*20+1, 20)
        y_ticks = np.arange(math.floor(y_min/20)*20, math.ceil(y_max/20)*20+1, 20)
        
        # 设置刻度
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
    
    # 设置标题和标签
    ax.set_title(f'Route {index+1}', fontsize=9)
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 添加统计信息
    orig_avg, orig_std, orig_min, orig_max = calculate_average_distance(points)
    red_avg, red_std, red_min, red_max = calculate_average_distance(reduced_points)
    
    info_text = (
        f'Pts: {len(points)} → {len(reduced_points)}\n'
        f'Avg: {orig_avg:.2f} → {red_avg:.2f}'
    )
    
    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5), fontsize=7)

# 可视化一批对比图（12个为一组）并保存
def visualize_batch_comparison(points_batch, output_dir="./output", batch_idx=1, 
                              nrows=3, ncols=4, figsize=(18, 14)):
    """
    将一批路线对比图排列成子图并保存
    
    参数:
    - points_batch: 一批路线点数据 (最多12条)
    - output_dir: 输出目录
    - batch_idx: 批次索引
    - nrows: 子图行数
    - ncols: 子图列数
    - figsize: 整个图形的尺寸
    """
    n_routes = len(points_batch)
    n_subplots = nrows * ncols
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图形
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # 如果只有一行，axes可能是一维数组，转换为二维
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = np.array([axes]).T
    
    # 展平axes以便迭代
    axes = axes.flatten()
    
    # 为每条路线绘制对比图
    for i in range(n_subplots):
        if i < n_routes:
            plot_comparison_on_axis(axes[i], points_batch[i], i)
        else:
            # 如果没有足够的路线，隐藏多余的子图
            axes[i].axis('off')
    
    # 移除未使用的子图坐标轴
    for j in range(n_routes, n_subplots):
        axes[j].axis('off')
    
    # 调整子图布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题腾出空间
    plt.suptitle(f'Route Comparison Batch {batch_idx} ({n_routes} routes)', fontsize=16)
    
    # 保存图形
    output_path = os.path.join(output_dir, f"route_comparison_batch_{batch_idx}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"批量对比图已保存到: {output_path}")
    return output_path

# 分批处理数据
def process_in_batches(points_by_line, batch_size=12, output_dir="./comparison_results"):
    total_lines = len(points_by_line)
    num_batches = math.ceil(total_lines / batch_size)
    
    print(f"共有 {total_lines} 行数据, 将分 {num_batches} 批处理 (每批最多 {batch_size} 条)")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_lines)
        batch = points_by_line[start_idx:end_idx]
        
        print(f"处理第 {i+1} 批, 包含 {len(batch)} 条路线 (索引 {start_idx+1} 到 {end_idx})")
        visualize_batch_comparison(batch, output_dir, i+1)

# 主函数
def main():
    file_path = "/home/zhanglingjun.zlj/code/Bench2Drive/train_bev-bases.jsonl"  # 替换为你的文件路径
    output_dir = "./visualization_results_1000"  # 输出目录
    
    # 读取数据
    points_by_line = read_points_from_jsonl(file_path)
    print(f"从文件中读取了 {len(points_by_line)} 行数据")
    
    # 过滤掉点数太少的路线
    filtered_points = [points for points in points_by_line if len(points) >= 5]
    print(f"过滤后保留 {len(filtered_points)} 条有效路线 (点数 >= 5)")
    
    # 处理所有路线
    process_in_batches(filtered_points, batch_size=12, output_dir=output_dir)

if __name__ == "__main__":
    main()
