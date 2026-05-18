import cv2
import os
import sys
import numpy as np
import json
from tqdm import trange


def create_video(images_folder, output_video, fps, font_scale, text_color, text_position):
    images = [img for img in os.listdir(os.path.join(images_folder, 'rgb_front')) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()

    frame = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for i in trange(0, len(images)*10, 10):
        image = images[i//10]
        f = open(os.path.join(images_folder, f'meta/{i:04}.json'), 'r')
        meta = json.load(f)
        steer = float(meta['steer'])
        throttle = float(meta['throttle'])
        brake = float(meta['brake'])
        # command = float(meta['command'])
        # command_list = ["VOID", "LEFT", "RIGHT", "STRAIGHT", "LANE FOLLOW", "CHANGE LANE LEFT",  "CHANGE LANE RIGHT",]
        speed = float(meta['speed'])
        text = f'speed: {round(speed,2)}, steer: {round(steer,2)}, throttle: {round(throttle,2)}, brake: {round(brake,2)}'#, command: {command_list[int(command)]}'
        img = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), image))
        cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
        video.write(img)
    video.release()

if __name__ == '__main__':
    
    res_path = sys.argv[1]
    output_path = sys.argv[2]
    os.makedirs(output_path, exist_ok=True)
    base_name = res_path.split('/')[-1]
    if base_name == 'Scenarios':
        images_folders = [os.path.join(res_path, folder) for folder in os.listdir(res_path)]
    else:
        images_folders = [res_path]
    for images_folder in images_folders:
        # 判断images_folder是否为文件夹
        if not os.path.isdir(images_folder):
            continue
        base_name = images_folder.split('/')[-1]
        output_video = os.path.join(output_path, f'{base_name}.mp4')
        fps = 10
        font_scale = 1
        text_color = (255, 255, 255)
        text_position = (50, 50)
        print(f'Creating video {output_video}')
        create_video(images_folder, output_video, fps, font_scale, text_color, text_position)
