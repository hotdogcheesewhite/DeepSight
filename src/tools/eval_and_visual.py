import json
import os
import re
import cv2
import random
import numpy as np
from tqdm import tqdm



def visual_for_crop(scene_path, visual_path, k=5):
    # 定义路径
    hz_index = list(range(0, 21, 5))
    scene_name = scene_path.split('/')[-1]
    bev_img_folders = [os.path.join(scene_path, 'camera', f'rgb_bev_{i}th-hz') for i in hz_index]
    img_files = os.listdir(bev_img_folders[0])
    img_files = [f for f in img_files if f.endswith('.jpg')]
    num_img = len(img_files)
    random_index = random.sample(list(range(num_img)), k)
    for index in random_index:
        visual_img = []
        for bev_img_folder in bev_img_folders:
            img = cv2.imread(os.path.join(bev_img_folder, f'{index:05d}.jpg'))
            visual_img.append(img)
        visual_img = np.concatenate(visual_img, axis=1)
        cv2.imwrite(os.path.join(visual_path, f'{scene_name}_{index:05d}.jpg'), visual_img)


def parse_prompt_and_answer(prompt, answer):
    # 解析 prompt
    # command = prompt.split('Mission Goal: ')[1].split(' ')[0]
    targetpoint = prompt.split('These are the target pixel tokens: ')[1].split(' ')[0]
    his_trajs = prompt.split('Historical trajectory: ')[1].split(' ')[0]
    speed_content = prompt.split('current speed info: ')[1].split('\n')[0]
    # 解析 answer
    future_trajs_pixel = answer.split('future pixel tokens: ')[1].split('. </answer>')[0]
    future_trajs = answer.split('future waypoints: ')[1].split('. </answer>')[0]

    # print('原始label', command, speed_content, his_trajs, future_trajs, future_trajs_pixel)

    # 轨迹转坐标
    pattern = r"[-+]?\d*\.\d+|[-+]?\d+"
    matches = re.findall(pattern, future_trajs)
    future_trajs = [[float(matches[i]), float(matches[i+1])] for i in range(0, len(matches), 2)]

    matches = re.findall(pattern, his_trajs)
    his_trajs = [[float(matches[i]), float(matches[i+1])] for i in range(0, len(matches), 2)]

    # 像素转坐标
    pattern = r'<\|pixel_token_([-+]?\d+)\|>'
    matches = re.findall(pattern, future_trajs_pixel)
    future_trajs_pixel = [[int(matches[i+1]), int(matches[i])] for i in range(0, len(matches), 2)]
    matches = re.findall(pattern, targetpoint)
    targetpointpixel = [[int(matches[i+1]), int(matches[i])] for i in range(0, len(matches), 2)]

    return speed_content, his_trajs, future_trajs, future_trajs_pixel, targetpointpixel


def parse_answer(answer):
    # 解析 answer
    future_trajs_pixel = answer.split('future pixel tokens: ')[1].split('. </answer>')[0]
    future_trajs = answer.split('future waypoints: ')[1].split('. </answer>')[0]

    # 轨迹转坐标
    pattern = r"[-+]?\d*\.\d+|[-+]?\d+"
    matches = re.findall(pattern, future_trajs)
    future_trajs = [[float(matches[i]), float(matches[i+1])] for i in range(0, len(matches), 2)]

    # 像素转坐标
    pattern = r'<\|pixel_token_([-+]?\d+)\|>'
    matches = re.findall(pattern, future_trajs_pixel)
    future_trajs_pixel = [[int(matches[i+1]), int(matches[i])] for i in range(0, len(matches), 2)]

    return future_trajs, future_trajs_pixel




def transform_traj_for_visual(his_trajs, future_trajs, future_trajs_pixel, targetpointpixel, pred_trajs=None, pred_trajs_pixel=None, cur_anno=None):
    his_trajs = transform_traj2pixel(his_trajs, cur_anno)
    future_trajs = transform_traj2pixel(future_trajs, cur_anno)
    future_trajs_pixel = transform_pixel2pixel(future_trajs_pixel)
    targetpointpixel = transform_pixel2pixel(targetpointpixel)
    if pred_trajs is not None:
        pred_trajs = transform_traj2pixel(pred_trajs, cur_anno)
    if pred_trajs_pixel is not None:
        pred_trajs_pixel = transform_pixel2pixel(pred_trajs_pixel)
    return his_trajs, future_trajs, future_trajs_pixel, pred_trajs, pred_trajs_pixel, targetpointpixel


def transform_traj2pixel(trajs, cur_anno):
    # 自车坐标系转相机坐标系
    trajs = np.array(trajs) # N, 2
    trajs = trajs[:, ::-1]
    trajs = np.concatenate([trajs, np.zeros((trajs.shape[0], 2))], axis=1) # N, 4
    # location_0 = cur_anno["bounding_boxes"][0]["location"]
    # extent = cur_anno["bounding_boxes"][0]["extent"]
    # location_0[2] = location_0[2] - extent[2] # 移动至地面
    trajs[:, 3] = 1
    trajs[:, 2] = trajs[:, 0]
    trajs[:, 0] = 0
    # trajs[:, -2] = location_0[2]
    # 相机坐标系转像素坐标系
    # world2ego = cur_anno["bounding_boxes"][0]["world2ego"]
    cam2ego = np.array(cur_anno["sensors"]["TOP_DOWN"]["cam2ego"])
    intrinsic = cur_anno["sensors"]["TOP_DOWN"]["intrinsic"]
    # print(intrinsic)
    ego2cam = np.linalg.inv(cam2ego)
    pixel_trajs = []
    for traj in trajs:
        # print(traj)
        traj_cam = cam2ego @ traj
        Xc, Yc, Zc = traj_cam[:3]
        fx, fy = intrinsic[0][0], intrinsic[1][1]
        cx, cy = intrinsic[0][2], intrinsic[1][2]
        u = fx * (Xc / Zc) + cx
        v = - fy * (Yc / Zc) + cy
        u, v = round(u), round(v)
        pixel_trajs.append((u, v))
        # print(traj.tolist(), traj_cam.tolist(),  (u, v))
    
    return pixel_trajs


def transform_pixel2pixel(trajs_pixel):
    # 相对坐标系转绝对坐标系
    for traj in trajs_pixel:
        traj[0] = 800 + traj[0] * 2
        traj[1] = 450 - traj[1] * 2

    return trajs_pixel


def visual_traj_on_bev(bev_img, his_trajs, future_trajs, future_trajs_pixel, pred_trajs=None, pred_trajs_pixel=None, targetpointpixel=None):
    for traj in his_trajs:
        cv2.circle(bev_img, tuple(traj), 10, (0, 0, 255), -1)

    for traj in future_trajs:
        cv2.circle(bev_img, tuple(traj), 10, (0, 255, 0), -1)

    for traj in future_trajs_pixel:
        cv2.circle(bev_img, tuple(traj), 5, (255, 0, 0), -1)
    
    if pred_trajs is not None:
        for traj in pred_trajs:
            cv2.circle(bev_img, tuple(traj), 6, (0, 0, 255), 2)
    
    if pred_trajs_pixel is not None:
        for traj in pred_trajs_pixel:
            cv2.circle(bev_img, tuple(traj), 3, (0, 255, 255), -1)
    
    for traj in targetpointpixel:
        cv2.circle(bev_img, tuple(traj), 10, (0, 255, 255), -1)
    
    return bev_img


def visual_for_bev(label, pred=None, visual_path=None):
    # {
    #     "messages": [  
    #         { "content": prompt, "role": "user" },
    #         { "content": answer,  "role": "assistant" }
    #     ],
    #     "images": images
    # }

    # 绘制文字和可视化点：
    prompt = label['messages'][0]['content']
    answer = label['messages'][1]['content']
    speed_content, his_trajs, future_trajs, future_trajs_pixel, targetpointpixel = parse_prompt_and_answer(prompt, answer)
    print('像素坐标', future_trajs_pixel, targetpointpixel)
    if pred is not None:
        pred_trajs, pred_trajs_pixel = parse_answer(pred['pred'])
    else:
        pred_trajs, pred_trajs_pixel = None, None
    # print('提取label', his_trajs, future_trajs, future_trajs_pixel)
    visual_text = f'current speed info: {speed_content}\nHistorical trajectory: {his_trajs}' + \
                  f'Future trajectory pixel: {future_trajs_pixel}\nFuture trajectory: {future_trajs}'
    cur_front_img = label['images'][4]
    # /mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-base/AccidentTwoWays_Town12_Route1102_Weather10/camera/rgb_front/00002.jpg
    cur_path_keys = cur_front_img.split('/')
    cur_anno = cur_path_keys[:-3] + ['anno', cur_path_keys[-1].replace('.jpg', '.json')]
    cur_anno = json.load(open('/'.join(cur_anno)))
    bev_img = cur_front_img.replace('rgb_front', 'rgb_top_down')
    his_trajs, future_trajs, future_trajs_pixel, pred_trajs, pred_trajs_pixel, targetpointpixel = transform_traj_for_visual(his_trajs, future_trajs, future_trajs_pixel, targetpointpixel, pred_trajs, pred_trajs_pixel, cur_anno)

    bev_img = cv2.imread(bev_img)
    bev_img = visual_traj_on_bev(bev_img, his_trajs, future_trajs, future_trajs_pixel, pred_trajs, pred_trajs_pixel, targetpointpixel)
    bev_img = cv2.resize(bev_img, dsize=None, fx=0.5, fy=0.5)

    dst_h, dst_w = 364, 644
    bev_h, bev_w = 256, 256
    # 创建画布：
    canvas = np.zeros((dst_h * 3 + bev_h * 2, dst_w * 4, 3), dtype=np.uint8)
    images = label['images']
     # 绘制历史帧图像
    for i in range(4):
        img = cv2.imread(images[i])
        img = cv2.resize(img, (dst_w, dst_h))
        canvas[:dst_h, dst_w * i:dst_w * (i + 1), :] = img
    # 绘制环视图
    indexs = [5, 4, 6, 8, 7, 9]
    for i, index in enumerate(indexs):
        img = cv2.imread(images[index])
        img = cv2.resize(img, (dst_w, dst_h))
        h_i, w_i = i // 3 + 1, i % 3
        canvas[dst_h * h_i:dst_h * (h_i + 1), dst_w * w_i:dst_w * (w_i + 1), :] = img
    for index in range(10, 15):
        # 绘制 BEV 图像
        target_img = images[index]
        target_img = cv2.imread(target_img)
        target_img = cv2.resize(target_img, (bev_w, bev_h))
        canvas[dst_h * 3 : dst_h * 3 + bev_h, bev_w * (index - 10) : bev_w * (index - 10) + bev_w, :] = target_img
    # 绘制文字和可视化BEV图
    cv2.putText(canvas, visual_text, (dst_h * 3 + bev_h + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    canvas[dst_h * 3 : dst_h * 3 + bev_img.shape[0], bev_w * 5 : bev_w * 5 + bev_img.shape[1], :] = bev_img
    cv2.imwrite(os.path.join(visual_path, cur_path_keys[-4] + '_' + cur_path_keys[-1] ), canvas)


def main_for_vis_train():
    train_json = '/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/create_data/train_bev-test.jsonl'
    # train_json = '/mnt/nas-data-1/wuchangjie.wcj/data/ad_ex2/train_bev-base.jsonl'
    visual_path = 'debug/visual_train-base'

    img_width, img_height = 1600, 900
    os.makedirs(visual_path, exist_ok=True)

    # 遍历 base_folder 下的每个子文件夹
    print(f'开始读取{train_json}')
    with open(train_json, 'r') as f:
        lines = f.readlines()
    print(f'读取{train_json}完成')

    k = min(100, len(lines))
    lines = random.sample(lines, k)
    for line in tqdm(lines):
        label = json.loads(line)
        visual_for_bev(label, None, visual_path)


def main_for_vis_infer():
    gt_json = '/mnt/nas-data-1/wuchangjie.wcj/data/ad_ex2/train_bev-test.jsonl'
    pred_json = 'debug/res_for_bev_ex3_final.json'
    visual_path = 'debug/visual_pred_ex3'

    os.makedirs(visual_path, exist_ok=True)
    # 遍历 base_folder 下的每个子文件夹
    print(f'开始读取{gt_json}')
    with open(gt_json, 'r') as f:
        lines = f.readlines()
    print(f'读取{gt_json}完成')

    print(f'开始读取{pred_json}')
    with open(pred_json, 'r') as f:
        lines_pred = f.readlines()
    print(f'读取{pred_json}完成')

    assert len(lines_pred) == len(lines), f'{len(lines_pred)} != {len(lines)}'

    k = min(100, len(lines))
    random_indexs = random.sample(list(range(len(lines))), k)
    for index in tqdm(random_indexs):
        label = json.loads(lines[index])
        pred = json.loads(lines_pred[index])
        visual_for_bev(label, pred, visual_path)


def print_l2_loss(losses_1s, losses_2s):
    # 转为 numpy 数组
    losses_1s = np.array(losses_1s)
    losses_2s = np.array(losses_2s)

    # 输出结果
    print(f"\n🎯 轨迹预测误差评估结果（仅 1s 和 2s）")
    print(f"{'时段':<6} {'样本数':<8} {'平均L2误差':<12} {'标准差':<10} {'最小值':<8} {'最大值':<8}")
    print("-" * 60)

    def print_stat(name, arr):
        if len(arr) > 0:
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            min_val = np.min(arr)
            max_val = np.max(arr)
            count = len(arr)
        else:
            mean_val = std_val = min_val = max_val = 0.0
            count = 0
        print(f"{name:<6} {count:<8} {mean_val:<12.6f} {std_val:<10.6f} {min_val:<8.6f} {max_val:<8.6f}")

    print_stat("1s", losses_1s)
    print_stat("2s", losses_2s)

    # 总体平均（只基于可用的1s和2s误差）
    all_losses = np.concatenate((losses_1s, losses_2s))
    overall_avg = np.mean(all_losses)
    print(f"\navg (overall): {overall_avg:.6f}")


def cal_l2_loss(future_trajs, pred_trajs):
    future_trajs = np.array(future_trajs)
    pred_trajs = np.array(pred_trajs)
    # print(future_trajs, pred_trajs)
    distances = np.sqrt(np.sum((future_trajs[:2] - pred_trajs[:2]) ** 2, axis=1))
    avg_loss_1s = np.mean(distances)
    distances = np.sqrt(np.sum((future_trajs - pred_trajs) ** 2, axis=1))
    avg_loss_2s = np.mean(distances)
    return avg_loss_1s, avg_loss_2s


def main_for_eval_l2():
    gt_json = '/mnt/nas-data-1/wuchangjie.wcj/data/ad_ex2/train_bev-test.jsonl'
    pred_json = 'debug/res_for_bev_ex3_final_v2.json'
    pred_json = 'debug/res_for_bev_ex3_full_final_v2.json'
    visual_path = 'debug/visual_pred_ex3_v2_full'
    visual_flag = False
    os.makedirs(visual_path, exist_ok=True)
    print(f'开始读取{gt_json}')
    with open(gt_json, 'r') as f:
        lines = f.readlines()
    print(f'读取{gt_json}完成')

    print(f'开始读取{pred_json}')
    with open(pred_json, 'r') as f:
        lines_pred = f.readlines()
    print(f'读取{pred_json}完成')

    assert len(lines_pred) == len(lines), f'{len(lines_pred)} != {len(lines)}'

    losses_1s, losses_2s = [], []
    for gt_l, pred_l in zip(lines, lines_pred):
        label = json.loads(gt_l)
        pred = json.loads(pred_l)
        prompt = label['messages'][0]['content']
        answer = label['messages'][1]['content']
        speed_content, his_trajs, future_trajs, future_trajs_pixel = parse_prompt_and_answer(prompt, answer)
        pred_trajs, pred_trajs_pixel = parse_answer(pred['pred'])
        # print(future_trajs, pred_trajs)
        avg_loss_1s, avg_loss_2s = cal_l2_loss(future_trajs, pred_trajs)
        if visual_flag and avg_loss_2s > 5:
            print(avg_loss_2s)
            visual_for_bev(label, pred, visual_path)
        losses_1s.append(avg_loss_1s)
        losses_2s.append(avg_loss_2s)
    print_l2_loss(losses_1s, losses_2s)



if __name__ == '__main__':
    # main_for_vis_infer()
    main_for_vis_train()
    # main_for_eval_l2()