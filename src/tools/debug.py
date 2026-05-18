import numpy as np
import json
import math
import cv2
import os



def transform_next_bev_img(info_0, info_1, img):

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
    print(u, v)

    # 获取朝向角
    theta_0 = info_0["theta"]
    theta_1 = info_1["theta"]
    rotate = (theta_0 - theta_1) % (2 * np.pi)
    angle = rotate * 180 / np.pi

    # 对图像进行旋转和平移变换
    height, width = img.shape[:2]
    center = (800, 450)
    img = cv2.circle(img, center, 5, (0, 0, 255), -1)
    img = cv2.circle(img, (int(u), int(v)), 5, (0, 255, 0), -1)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    translated_img = cv2.circle(rotated_img, (800, 450), 12, (0, 0, 255), 2)

    target_point = np.array([u, v, 1])
    target_point = rotation_matrix @ target_point

    shift_x = center[0] - target_point[0]
    shift_y = center[1] - target_point[1]
    print(shift_x, shift_y)
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_img = cv2.warpAffine(rotated_img, translation_matrix, (width, height))
    translated_img = cv2.circle(translated_img, (800, 450), 12, (0, 0, 255), 1)

    return img, rotated_img, translated_img


# for i in range(5, 11, 5):
if __name__ == '__main__':

    # visual_path = 'debug/visual-crop-bev'
    # if not os.path.exists(visual_path):
    #     os.makedirs(visual_path, exist_ok=True)
    # base_anno = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-val/SignalizedJunctionLeftTurnEnterFlow_Town13_Route657_Weather2/anno/00060.json'
    # cur_anno = json.load(open(base_anno, 'r'))
    # cur_img = cv2.imread('/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-val/SignalizedJunctionLeftTurnEnterFlow_Town13_Route657_Weather2/camera/rgb_top_down/00060.jpg')
    # cv2.circle(cur_img, (800, 450), 5, (0, 0, 255), -1)
    # for i in range(65, 80, 5):
    #     anno_file = f'/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-val/SignalizedJunctionLeftTurnEnterFlow_Town13_Route657_Weather2/anno/{i:05d}.json'
    #     with open(anno_file, 'r') as f:
    #         info_1 = json.load(f)
    #     bev_img_file = f'/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-val/SignalizedJunctionLeftTurnEnterFlow_Town13_Route657_Weather2/camera/rgb_top_down/{i:05d}.jpg'
    #     img = cv2.imread(bev_img_file)
    #     img, rotated_img, translated_img = transform_next_bev_img(cur_anno, info_1, img)
    #     img = np.concatenate([cur_img, img, rotated_img, translated_img], axis=0)
    #     cv2.imwrite(os.path.join(visual_path, f'{i:05d}.jpg'), img)

    world2cam = [
                [
                    7.024841124803061e-06,
                    0.003275694791227579,
                    -0.9999946355819702,
                    408.6690979003906
                ],
                [
                    -0.9999547600746155,
                    0.009512114338576794,
                    2.4134391424013302e-05,
                    2435.11376953125
                ],
                [
                    0.009512142278254032,
                    0.9999493956565857,
                    0.003275613533332944,
                    -2548.35888671875
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
    location =  [ 2459.2412109375,   2523.728271484375, 416.95556640625 ] # 相机位置
    location =  [ 2459.241455078125, 2523.89111328125,   366.9558410644531  ] # 汽车位置
    world2cam = np.array(world2cam)
    location = np.array(location + [1])
    pix_location = world2cam @ location
    print('相机坐标', pix_location)

    cam2world = np.linalg.inv(world2cam)
    inv_location = cam2world @ pix_location
    print('逆-世界坐标',inv_location)
    print('世界坐标',location)
    