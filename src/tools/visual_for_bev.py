import os
import random
import cv2
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


if __name__ == '__main__':


    base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-base'
    visual_path = 'debug/visual_crop-base'

    base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-full'
    visual_path = 'debug/visual_crop-full'

    base_folder = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-val'
    visual_path = 'debug/visual_crop-val'

    img_width, img_height = 1600, 900
    os.makedirs(visual_path, exist_ok=True)

    # 遍历 base_folder 下的每个子文件夹
    scene_names = os.listdir(base_folder)
    scene_names = [os.path.join(base_folder, name) for name in scene_names if name[0] != '.']
    scene_names = [name for name in scene_names if os.path.isdir(name)]
    k = min(100, len(scene_names))
    scene_names = random.sample(scene_names, k)
    for scene_path in tqdm(scene_names):
        visual_for_crop(scene_path, visual_path, 1)