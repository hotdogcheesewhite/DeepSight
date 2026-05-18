import os
from PIL import Image
import json
import numpy as np
import cv2
from tqdm import tqdm
import random
from multiprocessing import Pool
import time


def transform_next_bev_img(info_0, info_1, img):

    # 每像素对应的实际距离 (m/px)
    # meters_per_pixel_x, meters_per_pixel_y = 0.08925925, 0.08925925 # ~0.08926 # ~0.08926

    # world2ego = info_1["bounding_boxes"][0]["world2ego"]
    # world2ego = np.array(world2ego)
    world2cam = info_1['sensors']["TOP_DOWN"]["world2cam"]
    world2cam = np.array(world2cam)
    intrinsic = info_1['sensors']["TOP_DOWN"]["intrinsic"]
    intrinsic = np.array(intrinsic)

    location_0 = info_0["bounding_boxes"][0]["location"]
    extent = info_0["bounding_boxes"][0]["extent"]
    location_0 = np.array(location_0 + [1])
    location_0[2] = location_0[2] - extent[2] # 移动至地面
    location_0 =  world2cam @ location_0
    Zc, Xc, Yc = location_0[:3]
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    cx, cy = intrinsic[0][2], intrinsic[1][2]
    u = fx * (Xc / Zc) + cx
    v = - fy * (Yc / Zc) + cy
    # print(location_0, location_1, shift)

    # 获取朝向角
    theta_0 = info_0["theta"]
    theta_1 = info_1["theta"]
    rotate = (theta_0 - theta_1) % (2 * np.pi)
    angle = rotate * 180 / np.pi

    # 对图像进行旋转和平移变换
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    # 旋转变换
    target_point = np.array([u, v, 1])
    target_point = rotation_matrix @ target_point
    shift_x = center[0] - target_point[0]
    shift_y = center[1] - target_point[1]
    
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_img = cv2.warpAffine(rotated_img, translation_matrix, (width, height))

    return translated_img


def crop_and_save_img(img, save_name):

    height, width = img.shape[:2]
    crop_width, crop_height = 512, 512

    # 计算裁剪区域（居中水平，顶部开始垂直）
    left = (width - crop_width) // 2
    top = 85
    right = left + crop_width
    bottom = top + crop_height

    croped_img = img[top:bottom, left:right]
    cv2.imwrite(save_name, croped_img)


def fuc_crop_imgs(scene_path):
    # print(f'Processing {scene_path}')
    # 定义路径
    try:
        anno_path = os.path.join(scene_path, 'anno')
        rgb_top_down_folder = os.path.join(scene_path, 'camera', 'rgb_top_down')
        hz_index = list(range(0, 21, 5))
        bev_img_folders = [os.path.join(scene_path, 'camera', f'rgb_bev_{i}th-hz') for i in hz_index]
        for bev_img_folder in bev_img_folders:
            if not os.path.exists(bev_img_folder):
                os.makedirs(bev_img_folder, exist_ok=True)
        img_files = os.listdir(rgb_top_down_folder)
        img_files = [f for f in img_files if f.endswith('.jpg')]
        num_img = len(img_files)
        # 预先加载所有的anno和img
        all_imgs = []
        all_annos = []
        for i in range(num_img):
            anno_file = os.path.join(anno_path, f'{i:05d}.json')
            anno_gz = f'{anno_file}.gz'
            if os.path.exists(anno_gz):
                if os.path.exists(anno_file):
                    os.system(f"sudo rm {anno_file}")
                os.system(f"sudo gzip -d {anno_gz}")
            anno = json.load(open(anno_file))
            img = cv2.imread(os.path.join(rgb_top_down_folder, f'{i:05d}.jpg'))
            cv2.circle(img, (img_width // 2, img_height // 2), 10, (0, 150, 150), -1)
            assert img is not None
            all_imgs.append(img)
            all_annos.append(anno)

        for i in tqdm(range(num_img - 20)):
            cur_img = all_imgs[i]
            cur_anno = all_annos[i]
            crop_and_save_img(cur_img, os.path.join(bev_img_folders[0], f'{i:05d}.jpg'))
            for p_i, hz_i in enumerate(hz_index[1:]):
                next_img = all_imgs[i + hz_i]
                next_anno = all_annos[i + hz_i]
                transformed_img = transform_next_bev_img(cur_anno, next_anno, next_img)
                crop_and_save_img(transformed_img, os.path.join(bev_img_folders[p_i + 1], f'{i:05d}.jpg'))
    except Exception as e:
        print(scene_path,e)


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

if __name__ == '__main__':

    # base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-full'
    # base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-base'
    base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-val'
    img_width, img_height = 1600, 900

    # 遍历 base_folder 下的每个子文件夹
    scene_names = os.listdir(base_folder)
    scene_names = [os.path.join(base_folder, name) for name in scene_names if name[0] != '.']
    scene_names = [name for name in scene_names if os.path.isdir(name)]
    with Pool(processes=64) as pool:
        pool.map(fuc_crop_imgs, scene_names)

    # for scene_name in tqdm(scene_names):
    #     if scene_name[0] == '.':
    #         print(f'Skipping {scene_name}')
    #         continue
    #     scene_path = os.path.join(base_folder, scene_name)
        
    #     # 确保是目录
    #     if not os.path.isdir(scene_path):
    #         print(f'Skipping {scene_name}')
    #         continue
        
    #     # print(f'Processing {scene_path}')
    #     fuc_crop_imgs(scene_path)
    #     # assert 1==2

    
    # visual
    # visual_path = 'debug/visual_crop'
    # os.makedirs(visual_path, exist_ok=True)
    # for scene_name in os.listdir(base_folder):
    #     if scene_name[0] == '.':
    #         continue
    #     scene_path = os.path.join(base_folder, scene_name)
    #     visual_for_crop(scene_path, visual_path, 10)
    #     assert 1==2