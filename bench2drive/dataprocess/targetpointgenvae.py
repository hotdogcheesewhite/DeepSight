import json
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager
import copy
import math
import re

RESULT_MAP = None
RESULT_JSONL_PATH = '/home/zhanglingjun.zlj/code/road2rome/gt_indices_train_fulljsonl.jsonl'  # 替换为实际路径

def load_result_map():
    """加载result映射字典到全局变量"""
    global RESULT_MAP
    if RESULT_MAP is None:
        print(f"Loading result map from {RESULT_JSONL_PATH}...")
        RESULT_MAP = {}
        with open(RESULT_JSONL_PATH, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # 确保id是字符串类型
                    img_id = str(data['id'])
                    RESULT_MAP[img_id] = data['result']
                except Exception as e:
                    print(f"Error parsing line: {line}, {e}")
        print(f"Loaded {len(RESULT_MAP)} results into result map.")
    return RESULT_MAP

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

def get_images(i, history_path, suround_view_path, bev_img_folders):
    images = []
    for index in range(4, 0, -1):
        his_index = i - index * 5
        if his_index < 0:
            img_file = '/mnt/nas-data-1/zhanglingjun.zlj1/ad_data_process/sft_data_api_explain/hisblack.jpg'
        else:
            img_file = os.path.join(history_path, f'{his_index:05d}.jpg')
        assert os.path.exists(img_file), f"{img_file} not exists"
        images.append(img_file)
    # 环视图图像
    for cam_path in suround_view_path:
        img_file = os.path.join(cam_path, f'{i:05d}.jpg')
        assert os.path.exists(img_file), f"{img_file} not exists"
        images.append(img_file)
    # bev 图像
    for bev_img_folder in bev_img_folders:
        bev_img = os.path.join(bev_img_folder, f'{i:05d}.jpg')
        assert os.path.exists(bev_img), f"{bev_img} not exists"
        images.append(bev_img)

    assert len(images) == 15, f"{scene_path} {i} {len(images)}"
    return images


def get_prompt(command, his_trajs=None, speed_content=None, bevtargetpoints=None):
    history_prompt = "These are the vehicle's CAM_FRONT historical images: 2.0s ago <image> 1.5s ago <image> 1.0s ago <image> 0.5s ago <image>."
    soround_prompt = "These are the vehicle's current frame six-view images: CAM_FRONT:<image> CAM_FRONT_LEFT:<image> CAM_FRONT_RIGHT:<image> CAM_BACK:<image> CAM_BACK_LEFT:<image> CAM_BACK_RIGHT:<image>."
    # state_promt = f"Mission Goal: {command} Historical trajectory: {his_trajs} current speed info: {speed_content}"
    state_promt = f"These are the target pixel tokens: {bevtargetpoints} Historical trajectory: {his_trajs} current speed info: {speed_content}"
    # instruct_promt1 = f"<CoT_flag_{FLAGE}>"
    instruct_promt = "Based on the provided particulars, please generate BEV image and plan waypoints (0.5s intervals) for the next 2 seconds.\n"
    # print(type(history_prompt), type(soround_prompt), type(state_promt), type(instruct_promt))
    prompt = '\n'.join([history_prompt, soround_prompt, state_promt,instruct_promt])
    return prompt


def get_answer(think_content=None, bev_content=None, future_trajs_pixel=None, future_trajs=None,result = None):
    think_answer = "<think>These are the BEV image "
    pixel_answer = f"<answer> These are the future pixel tokens: {future_trajs_pixel}. </answer>"
    traj_answer = f"<answer> These are the future waypoints: {future_trajs}. </answer>"
    result1 = str(result)
    think_answerend = "</think>"
    result2 = think_answer+result1+think_answerend
    answer = '\n'.join([result2,pixel_answer, traj_answer])
    return answer


def format_trajs(trajs):
    str_trajs = []
    for traj in trajs:
        x, y = traj
        x = f'{x:.2f}'
        y = f'{y:.2f}'
        if x == '-0.00':
            x = '0.00'
        if y == '-0.00':
            y = '0.00'
        traj = f'({x},{y})'
        str_trajs.append(traj)
    str_trajs = ','.join(str_trajs)
    return f'[{str_trajs}]'


def parse_anno(index, all_annos, sampled_points):
    # 获取控制命令
    xtarget = all_annos[index]['x_command_near']
    ytarget = all_annos[index]['y_command_near']
    ztarget = all_annos[index]['bounding_boxes'][0]["location"][2]
    extent = all_annos[index]["bounding_boxes"][0]["extent"][2]
    ztarget = ztarget - extent
    command = all_annos[index]['next_command']
    # 获取速度信息
    speed_content = f'speed: {all_annos[index]["speed"]:.2f}, acceleration: {all_annos[index]["acceleration"][0]:.2f}'
    thick_content = None
    # 获取历史轨迹
    world2ego = np.array(all_annos[index]['bounding_boxes'][0]['world2ego'])
    world2cam = np.array(all_annos[index]["sensors"]["TOP_DOWN"]['world2cam'])
    intrinsic = np.array(all_annos[index]['sensors']["TOP_DOWN"]["intrinsic"])
    his_trajs = []
    for i in range(1, 5):
        his_index = index - i * 5
        if his_index < 0:
            if len(his_trajs) == 0:
                his_traj = (0, 0)
            else:
                his_traj = copy.deepcopy(his_trajs[0])
        else:
            his_traj = all_annos[his_index]['bounding_boxes'][0]["location"]
            his_traj = world2ego @ np.array(his_traj + [1])
            his_traj = tuple(his_traj[:2].tolist())
        his_trajs.insert(0, his_traj)
    his_trajs = format_trajs(his_trajs)
    # 获取未来轨迹标签
    future_trajs = []
    for i in range(1, 5):
        future_index = index + i * 5
        future_traj = all_annos[future_index]['bounding_boxes'][0]["location"]
        future_traj = world2ego @ np.array(future_traj + [1])
        future_traj = tuple(future_traj[:2].tolist())
        future_trajs.append(future_traj)
    future_trajs = format_trajs(future_trajs)
    # 获取未来轨迹像素标签
    future_trajs_pixel = []
    for i in range(1, 5):
        future_index = index + i * 5
        future_traj = all_annos[future_index]['bounding_boxes'][0]["location"]
        extent = all_annos[future_index]["bounding_boxes"][0]["extent"]
        future_traj = np.array(future_traj + [1])
        future_traj[2] = future_traj[2] - extent[2] # 移动至地面
        future_traj =  world2cam @ future_traj
        Zc, Xc, Yc = future_traj[:3]
        fx, fy = intrinsic[0][0], intrinsic[1][1]
        cx, cy = intrinsic[0][2], intrinsic[1][2]
        u = fx * (Xc / Zc) + cx
        v = - fy * (Yc / Zc) + cy
        dx, dy = round((u - 800)/2), round((450 - v)/2)
        dx, dy = max(-255, min(255, dx)), max(-255, min(255, dy))
        dx, dy = f'<|pixel_token_{dx}|>',f'<|pixel_token_{dy}|>'
        future_trajs_pixel.append(f'({dy},{dx})')
    future_trajs_pixel = ','.join(future_trajs_pixel)
    future_trajs_pixel = f'[{future_trajs_pixel}]'

    # 获取 bev content
    t, h, w, patchsize, n_cls, n_register = 5, 256, 256, 16, 1, 4
    l = t * (h * w // (patchsize ** 2) + n_cls + n_register)
    bev_content = []
    for i in range(l):
        bev_content.append(f"<|bev_token_{i}|>")
    bev_content = ''.join(bev_content)

    bevtargetpoints = []
    for point in sampled_points:
        points_3d = [point[0],point[1], ztarget]
        points_4d = np.array(points_3d + [1])
        bevtarget = world2cam @ points_4d
        Zc, Xc, Yc = bevtarget[:3]
        fx, fy = intrinsic[0][0], intrinsic[1][1]
        cx, cy = intrinsic[0][2], intrinsic[1][2]
        u = fx * (Xc / Zc) + cx
        v = - fy * (Yc / Zc) + cy
        dx, dy = round((u - 800)/2), round((450 - v)/2)
        # dx, dy = max(-255, min(255, dx)), max(-255, min(255, dy))
        if dy < -255 or dy > 255 or dx < -255 or dx > 255:
            continue  # 跳过当前时间点，不添加到结果
        dx, dy = f'<|pixel_token_{dx}|>',f'<|pixel_token_{dy}|>'
        bevtargetpoints.append(f'({dy},{dx})')
    bevtargetpoints = ','.join(bevtargetpoints)
    bevtargetpoints = f'[{bevtargetpoints}]'





    return his_trajs, speed_content, command, thick_content, bev_content, future_trajs_pixel, future_trajs, xtarget, ytarget, ztarget, bevtargetpoints

###############################################################################################################

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

def plot_sampled_points(waypoints, angle_threshold=15, 
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

##############################################################################################################
def extract_decision_and_summary(result):
    """
    从result字符串中提取决策判断和推理总结
    
    参数:
        result: 原始result字符串
        
    返回:
        tuple: (decision, summary) 
               decision: "是复杂场景" 部分
               summary: 推理总结部分
    """
    # 正则表达式模式：
    # 1. 匹配决策部分：\[根据路况信息判断是否需要复杂决策\]:\s*([^\n]+)
    # 2. 匹配总结部分：\[推理结果进行总结\]:\s*([\s\S]+)
    pattern = r'\[根据路况信息判断是否需要复杂决策\]:\s*([^\n]+)\s*\[推理结果进行总结\]:\s*([\s\S]+)'
    
    match = re.search(pattern, result)
    
    if match:
        decision = match.group(1).strip()  # 提取决策部分并去除首尾空格
        summary = match.group(2).strip()   # 提取总结部分并去除首尾空格
        
        # 特殊处理：移除决策部分末尾的句号（如果存在）
        if decision.endswith('。'):
            decision = decision[:-1]
        
        return decision, summary
    else:
        # 如果匹配失败，返回原始字符串或空值
        # print("Warning: Pattern not matched in result string")
        return "", ""


def create_train_json(scene_path, result_map):
    print(f"Processing {scene_path}")
    # 设置文件路径：
    anno_path = os.path.join(scene_path, 'anno')
    hz_index = list(range(0, 21, 5))
    bev_img_folders = [os.path.join(scene_path, 'camera', f'rgb_bev_{i}th-hz') for i in hz_index]
    history_path = os.path.join(scene_path,'camera', 'rgb_front')
    suround_view_path = ['front', 'front_left', 'front_right', 'back', 'back_left', 'back_right']
    suround_view_path = [os.path.join(scene_path, 'camera', f'rgb_{cam}') for cam in suround_view_path]
    # 遍历获取内容
    anno_files = os.listdir(anno_path)
    anno_files = [f for f in anno_files if f.endswith('.json')]
    all_annos = [json.load(open(os.path.join(anno_path, f))) for f in anno_files]
    
    nums = len(anno_files)
    res = []

    target_points = []

    for i in range(1, nums-20):
        xtarget = all_annos[i]['x_command_near']
        ytarget = all_annos[i]['y_command_near']
        
        new_point = (xtarget, ytarget)
        if new_point not in target_points:
            target_points.append(new_point)
    try:
        xtarget = all_annos[nums-20]['x_command_far']
        ytarget = all_annos[nums-20]['y_command_far']
        new_point = (xtarget, ytarget)     
        if new_point not in target_points:
            target_points.append(new_point)
    except IndexError:
        print(f"无法访问 all_annos[{nums-20}]，跳过此目标点")
    
    reduced_points, removed_indices = extract_straight_lanes(target_points, angle_threshold=0.05)
    sampled_points = plot_sampled_points(reduced_points, angle_threshold=15, max_extension=25, label='Route 1')
    

    for i in range(1, nums-20):
        # try:
             # 解析训练标注
            # import pdb; pdb.set_trace()
        his_trajs, speed_content, command, thick_content, bev_content, future_trajs_pixel, future_trajs, xtarget, ytarget, ztarget, bevtargetpoints = parse_anno(i, all_annos, sampled_points)
        

        images = get_images(i, history_path, suround_view_path, bev_img_folders)  # 获取图像
        bev_image_path = images[-1]  # 最后一个元素是bev图像
        result = result_map.get(str(bev_image_path), "")
        # print(result)
        # FLAGE = 'True'
        # if result =='':
        #     FLAGE = 'False'
        # decision, summary = extract_decision_and_summary(result)
        # import pdb; pdb.set_trace()
        prompt = get_prompt(command=command, his_trajs=his_trajs, speed_content=speed_content, bevtargetpoints = bevtargetpoints)     # 获取 prompt
        answer = get_answer(think_content=thick_content, bev_content=bev_content, future_trajs_pixel=future_trajs_pixel, future_trajs=future_trajs, result = result)  # 获取answer
        format_label = {
            "messages": [  
                { "content": prompt, "role": "user" },
                { "content": answer,  "role": "assistant" }
            ],
            "images": images[:-5]
        }
        res.append(json.dumps(format_label,ensure_ascii=False))
            # new_point = (xtarget, ytarget)
            # if new_point not in target_points:
            #     target_points.append(new_point)
        # except Exception as e:
        #     print(f"Error: {scene_path} {i}: {e}")
        #     continue
    base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-full'
    prefix_len = len(base_folder) + 1 
    relative_scene_names = scene_path[prefix_len:]
    os.makedirs(relative_scene_names, exist_ok=True)
    # target_points_json = json.dumps(target_points)

    with open(os.path.join(relative_scene_names, sub_train_json), 'w', encoding='utf-8') as f:
        f.write('\n'.join(res))

    # with open(os.path.join(relative_scene_names, sub_train_json), 'w',encoding='utf-8') as f:
    #     f.write('\n'.join(target_points))
def init_worker():
    """初始化工作进程，加载result_map"""
    global RESULT_MAP
    if RESULT_MAP is None:
        RESULT_MAP = load_result_map()

if __name__ == '__main__':
    RESULT_JSONL_PATH = '/home/zhanglingjun.zlj/code/road2rome/gt_indices_train_fulljsonl.jsonl'  # 必须替换为实际路径
    # base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-val'
    # train_json = '/home/zhanglingjun.zlj/code/Bench2Drive/train_bev-test.jsonl'
    # sub_train_json = 'train_bev_v3.jsonl'

    base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-full'
    train_json = '/home/zhanglingjun.zlj/code/Bench2Drive/train_bev-large0107.jsonl'
    sub_train_json = 'train_bev_v3.jsonl'

    # base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-val'
    # train_json = '/home/zhanglingjun.zlj/code/Bench2Drive/train_bev-test2.jsonl'
    # len_target_json = '/home/zhanglingjun.zlj/code/Bench2Drive/targetlength.jsonl'
    # sub_train_json = 'train_bev_v3.jsonl'
    # 遍历 base_folder 下的每个子文件夹
    scene_names = os.listdir(base_folder)
    scene_names = [os.path.join(base_folder, name) for name in scene_names if name[0] != '.']
    scene_names = [name for name in scene_names if os.path.isdir(name)]
    with Manager() as manager:
        result_map = manager.dict(load_result_map())
        with Pool(processes=64, initializer=init_worker) as pool:
            tasks = [(scene_path, result_map) for scene_path in scene_names]
            pool.starmap(create_train_json, tasks)
            # pool.map(create_train_json, scene_names)
    # for scene_path in tqdm(scene_names):
    #     create_train_json(scene_path)
    prefix_len = len(base_folder) + 1  # +1 是为了去掉路径分隔符 '/'
    relative_scene_names = [path[prefix_len:] for path in scene_names]
    
    with open(train_json, 'w', encoding='utf-8') as f:
        for scene_path in tqdm(relative_scene_names):
            f.write(open(os.path.join(scene_path, sub_train_json), encoding='utf-8').read())
            # f.write(open(sub_train_json), encoding='utf-8').read())
            f.write('\n')