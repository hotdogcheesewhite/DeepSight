import json
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import copy


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


def get_prompt(command, his_trajs=None, speed_content=None):
    history_prompt = "These are the vehicle's CAM_FRONT historical images: 2.0s ago <image> 1.5s ago <image> 1.0s ago <image> 0.5s ago <image>."
    soround_prompt = "These are the vehicle's current frame six-view images: CAM_FRONT:<image> CAM_FRONT_LEFT:<image> CAM_FRONT_RIGHT:<image> CAM_BACK:<image> CAM_BACK_LEFT:<image> CAM_BACK_RIGHT:<image>."
    state_promt = f"Mission Goal: {command} Historical trajectory: {his_trajs} current speed info: {speed_content}"
    instruct_promt = "Based on the provided particulars, please generate BEV image and plan waypoints (0.5s intervals) for the next 2 seconds.\n"
    # print(type(history_prompt), type(soround_prompt), type(state_promt), type(instruct_promt))
    prompt = '\n'.join([history_prompt, soround_prompt, state_promt, instruct_promt])
    return prompt


def get_answer(think_content=None, bev_content=None, future_trajs_pixel=None, future_trajs=None):
    
    think_answer = f"<think> {think_content} </think>"
    bev_answer = f"<|start_bev_token|>{bev_content}<|end_bev_token|>"
    pixel_answer = f"<answer> These are the future pixel tokens: {future_trajs_pixel}. </answer>"
    traj_answer = f"<answer> These are the future waypoints: {future_trajs}. </answer>"

    answer = '\n'.join([think_answer, bev_answer, pixel_answer, traj_answer])
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


def parse_anno(index, all_annos):
    # 获取控制命令
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

    return his_trajs, speed_content, command, thick_content, bev_content, future_trajs_pixel, future_trajs


def create_train_json(scene_path):
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
    textprompt = """1. 目标:
你正在复杂的城市交通环境中控制一辆自动驾驶车辆，当前可获得：六个摄像头视角图像（包括前、后、左、右、左前、右前），自我车辆过去2s轨迹，以及CAM_FRONT摄像头过去2秒的历史图像。你的任务是结合驾驶任务目标，规划自车在未来2s内的安全且合理的行驶行为。请使用中文进行推理与输出。

2. 过程概述:
请严格按照以下两个步骤依次进行推理：
a. [根据历史信息推理当前动作]
   根据像素特征，关注车道线、参照物等信息，判断当前车辆行为正处于以下哪个阶段：
   [跟车, 换道, 向左转弯, 向右转弯, 停车等待, 其他]
   输出要求：
   - 严格在提供的范围内选择，格式为“当前车辆行为正处于[阶段]”。（例：“当前车辆行为正处于换道阶段”。）
   - 若无法匹配列表，选择“其他”。

b. [根据路况信息判断是否需要复杂决策]
   若当前环境存在多目标交互（如路口通行、超越障碍车辆、汇入主路、避让紧急车辆等），需判断是否属于“复杂决策场景”。判断标准：
   - 若位于十字路口且自车行驶方向为绿灯、自车车道前方无车 → **优先考虑加速**；
   - 若前车尾灯持续闪烁（非转向灯）且无其他异常 → 可能为故障车辆，**视为复杂场景**；
   - 若位于无信号灯十字路口需转弯或通过 → **视为复杂场景**；
   - 若前方存在“STOP”标志牌且周围有潜在风险（如行人、交叉车流） → **视为复杂场景**；
   - 其他情况根据驾驶常识判断：无交互风险时为简单场景，否则为复杂场景。
   输出要求：
   - 仅输出“是复杂场景”或“是简单场景”。（例：“是复杂场景”）

c. [推理结果进行总结]
   综合上述推理结果，从候选动作集合中挑选安全合理的未来驾驶动作：
   - 方向控制（五选一）：[保持当前车道, 向左变道, 向右变道, 左转, 右转]
   - 速度控制（五选一）：[停车等待, 减速, 保持当前速度, 加速, 借道超车]
   输出要求：
   - 若为简单场景：直接输出动作组合，格式为“方向控制, 速度控制”（例：“保持当前车道, 减速”）。
   - 若为复杂场景：先说明原因，再输出动作组合，格式为“由于[原因]，[方向控制, 速度控制]”（例：“由于前方STOP标志牌且有行人通过，停车等待”）。

3. 输出格式示例：
[根据历史信息推理当前动作]:
当前车辆行为正处于跟车阶段。
[根据路况信息判断是否需要复杂决策]:
是简单场景。
[推理结果进行总结]:
保持当前车道, 保持当前速度  
    """
    COMMAND_DICT = {
        -1: '无',
        1: '左转',
        2: '右转',
        3: '直行',
        4: '保持当前车道',
        5: '向左变道',
        6: '向右变道',
    }
    for i in range(1, nums-20):
        try:
             # 解析训练标注
        #     # import pdb; pdb.set_trace()
            his_trajs, speed_content, command, thick_content, bev_content, future_trajs_pixel, future_trajs = parse_anno(i, all_annos)
            images = get_images(i, history_path, suround_view_path, bev_img_folders)  # 获取图像
            # prompt = get_prompt(command=command, his_trajs=his_trajs, speed_content=speed_content)     # 获取 prompt
            # answer = get_answer(think_content=thick_content, bev_content=bev_content, future_trajs_pixel=future_trajs_pixel, future_trajs=future_trajs)  # 获取answer
            textprompt1 = "当前驾驶任务目标: " + COMMAND_DICT[command] + "\n"
            format_label = {
                # "messages": [  
                #     { "content": prompt, "role": "user" },
                #     { "content": answer,  "role": "assistant" }
                # ],
                "prompt": textprompt1+textprompt,
                "images": images
            }
            res.append(json.dumps(format_label,ensure_ascii=False))
        except Exception as e:
            print(f"Error: {scene_path} {i}: {e}")
            continue

    with open(os.path.join(scene_path, sub_train_json), 'w',encoding='utf-8') as f:
        f.write('\n'.join(res))


if __name__ == '__main__':

    # base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-base'
    # train_json = '/mnt/nas-data-1/wuchangjie.wcj/data/ad_ex2/train_bev-base.jsonl'
    # sub_train_json = 'train_bev_v2.jsonl'

    # base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-full'
    # train_json = '/mnt/nas-data-1/wuchangjie.wcj/data/ad_ex2/train_bev-large.jsonl'
    # sub_train_json = 'train_bev_v2.jsonl'

    base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-val'
    train_json = '/mnt/nas-data-1/zhanglingjun.zlj1/Bench2Drive/train_bev-test1.jsonl'
    sub_train_json = 'train_bev_v3.jsonl'
    # 遍历 base_folder 下的每个子文件夹
    scene_names = os.listdir(base_folder)
    scene_names = [os.path.join(base_folder, name) for name in scene_names if name[0] != '.']
    scene_names = [name for name in scene_names if os.path.isdir(name)]

    with Pool(processes=64) as pool:
        pool.map(create_train_json, scene_names)
    # for scene_path in tqdm(scene_names):
    #     create_train_json(scene_path)

    with open(train_json, 'w', encoding='utf-8') as f:
        for scene_path in tqdm(scene_names):
            f.write(open(os.path.join(scene_path, sub_train_json), encoding='utf-8').read())
            f.write('\n')