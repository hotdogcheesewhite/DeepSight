import pickle
import re
import json
import argparse
import tiktoken
from nuscenes.nuscenes import NuScenes
from prompt_message import  generate_user_message, generate_assistant_message, generate_goal_message
import os
from pathlib import Path
from tqdm import tqdm
import traceback

def load_bev_tokens(front_img_path, img_tokens, offsets=[0, 5, 10, 15]):
    def to_token_str(nums):
        return ''.join(f'<|{num}|>' for num in nums)

    tokens = []
    for offset in offsets:
        path = convert_to_bev_path_with_offset(front_img_path, offset=offset)
        if path not in img_tokens:
            raise KeyError(f"Not found: {path}")
        tokens.append(to_token_str(img_tokens[path]['CAM_BEV']))
    return tokens

def convert_to_bev_path_with_offset(img_path, offset=0):
    # 使用 Path 解析路径
    p = Path(img_path)
    
    # 提取文件名中的数字（如 00000）
    match = re.search(r'(\d+)\.jpg$', p.name)
    if not match:
        raise ValueError(f"无法从文件名中提取帧号: {p.name}")
    
    frame_num = int(match.group(1))         # 转为整数
    new_frame_num = frame_num + offset      # 加上偏移（比如 +5）
    new_filename = f"{new_frame_num:05d}.jpg"  # 格式化为 000005 这种形式

    # 构造新路径：替换目录名为 rgb_bev，替换文件名为新编号
    new_parent = str(p.parent).replace("rgb_back_right", "rgb_bev")
    
    return str(Path(new_parent) / new_filename)


def load_trajtoken_dict(json_file_path):
    trajtoken_dict = {}
    with open(json_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 归一化路径，避免 /path 和 //path 不一致问题
                img_path = os.path.normpath(data["image_filename"])
                trajtoken_dict[img_path] = data["trajtoken"]
            except Exception as e:
                print(f"解析行失败: {e}")
                continue
    return trajtoken_dict

# === 预加载一次 ===
json_path = '/mnt/nas-data-1/zhanglingjun.zlj1/ad_data_process/abench2drive/converted/bevpixel1.json'
trajtoken_map = load_trajtoken_dict(json_path)
system="You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0). Units: meters. Based on the provided particulars, please output the CAM_FRONT image at the 0.5 second in the future and plan waypoints (0.5s intervals) for the next 3 seconds."

parser = argparse.ArgumentParser(description="Choose to use train or val tokens.")
parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Select 'train' or 'val' token set")
args = parser.parse_args()

data = pickle.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/data/nuscene/data/cached_nuscenes_info.pkl', 'rb'))
split = json.load(open('/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/create_data/full_split.json', 'r'))
tokens = split[args.split]

num_train_samples = len(tokens)
train_ratio = 1

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_language_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0
traj_only = True

with open('/mnt/nas-data-1/zhanglingjun.zlj1/data/reasonplan/train_final_1022.json', 'r') as file:#trains or val
# with open('/mnt/nas-data-1/zhanglingjun.zlj1/data/reasonplan/train_final_0912.json', 'r') as file:#trains or val
# with open('/mnt/nas-data-1/zhanglingjun.zlj1/ad_data_process/sft_data_api_explain/LLaMA-Factory-data/train_v3_ablation_0905.json', 'r') as file:
    full_train_token_prompt = json.load(file)

# with open('/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/MoVQGAN/MoVQGAN/gt_indices_bench2drive_bev.jsonl', 'r') as file:#trains or val
#     img_tokens = json.load(file)
img_tokens = {}
with open('/mnt/nas-data-1/zhanglingjun.zlj1/FSDrive-main/MoVQGAN/MoVQGAN/gt_indices_bench2drive_bev.jsonl', 'r') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        img_tokens.update(data) 

train_messages = []
prompt1 = "1.目标:\n你正在复杂的城市交通环境中控制一辆自动驾驶车辆，当前可获得：六个摄像头视角图像，自我车辆过去2s轨迹，以及<CAM_FRONT>过去2秒的历史图像。你的任务是根据导航目标，结合感知信息，生成未来0.5s的前视图图像，并规划自车在未来3s内的安全且合理的轨迹。2.使用这样的标准格式：<think>[图像]</think>\n<answer>[轨迹文本]<answer>"
prompt1 = ""
base_path = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-base'#base or val
# base_path = '/mnt/nas-data-1/zhanglingjun.zlj1/data/bench2drive-val'#base or val
for i, listnumber in enumerate(tqdm(full_train_token_prompt, desc="Processing images")):

    id = listnumber['frame_id']
    his_path = listnumber['history_frames']
    img_path = listnumber['image_path']
    all_paths = []
    final_path = []
    all_paths.extend(his_path)
    for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        if cam in img_path:
            all_paths.append(img_path[cam])
    cmd_str = listnumber['cmd_str']
    future_traj_str = listnumber['future_traj_str']
    # print(future_traj_str)
    coordinates = re.findall(r'\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)', future_traj_str)
    coordinates = [(float(x), float(y)) for x, y in coordinates]
    coordinates = coordinates[:4]
    filtered_str = ", ".join([f"({x:.2f}, {y:.2f})" for x, y in coordinates])

    his_traj = listnumber['history_trajectory']
    answer = listnumber['answer']
    sysprompt = listnumber['system_prompt']
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, answer, re.DOTALL)
    think_content = match.group(1)

    pattern2 = r"the current vehicle's longitudinal speed is [-+]?\d+\.?\d* m/s, and the longitudinal acceleration is [-+]?\d+\.?\d* m/s\^2"
    match2 = re.search(pattern2,sysprompt)
    speed_content = match2.group()

    


    for item in all_paths:
        if item.startswith('v1/'):
            new_path = os.path.join(base_path, item[3:])
            final_path.append(new_path)
        else:
            new_path = item
            final_path.append(new_path)
    
    try:
        front_img_path=final_path[-1]
    # bev_img_path = convert_to_bev_path_with_offset(front_img_path, offset=0)

    # print(bev_img_path)
    # next_img_token=img_tokens[bev_img_path]['CAM_BEV']
    # next_img_token = str(next_img_token).replace(" ", "")
    # numbers = next_img_token.strip('[]').split(',')
    # next_img_token = ''.join([f'<|{num}|>' for num in numbers])
        tokens = load_bev_tokens(front_img_path, img_tokens)
        next_img_token, next_img_token1, next_img_token2, next_img_token3 = tokens
    except:
        print("failed")
        continue

    front_img_path=final_path[-1]
    normalized_path = os.path.normpath(front_img_path)
    trajtoken = trajtoken_map.get(normalized_path, None)

    train_data_one = {
                    "messages": [
                        {
                            # "content": "These are the vehicle's current frame six-view images: CAM_FRONT:<image>\nCAM_FRONT_LEFT:<image>\nCAM_FRONT_RIGHT:<image>\nCAM_BACK:<image>\nCAM_BACK_LEFT:<image>\nCAM_BACK_RIGHT:<image>\n"+"Mission Goal:"+cmd_str+"\n"+"Historical trajectory: " + his_traj +". Based on the provided particulars, please plan waypoints (0.5s intervals) for the next 3 seconds.",
                            # "content": "These are the vehicle's CAM_FRONT historical images: 2.0s ago<image>\n1.5s ago<image>\n1.0s ago<image>\n0.5s ago<image>\n"+"These are the vehicle's current frame six-view images: CAM_FRONT:<image>\nCAM_FRONT_LEFT:<image>\nCAM_FRONT_RIGHT:<image>\nCAM_BACK:<image>\nCAM_BACK_LEFT:<image>\nCAM_BACK_RIGHT:<image>\n"+"Mission Goal:"+cmd_str+"\n"+"Historical trajectory: " + his_traj +", "+speed_content+". Based on the provided particulars, please output the BEV image at the 0.5 second in the future and plan waypoints (0.5s intervals) for the next 2 seconds.Please output according to this format:<think>[Reasoning text][Img token]</think><answer>[Trajectory text]</answer>\n",
                            # "content": "These are the vehicle's CAM_FRONT historical images: 2.0s ago<image>\n1.5s ago<image>\n1.0s ago<image>\n0.5s ago<image>\n"+"These are the vehicle's current frame six-view images: CAM_FRONT:<image>\nCAM_FRONT_LEFT:<image>\nCAM_FRONT_RIGHT:<image>\nCAM_BACK:<image>\nCAM_BACK_LEFT:<image>\nCAM_BACK_RIGHT:<image>\n"+"Mission Goal:"+cmd_str+"\n"+"Historical trajectory: " + his_traj +", "+speed_content+". Based on the provided particulars, please generate 4 BEV images (0.5s intervals) from the current moment and plan waypoints (0.5s intervals) for the next 2 seconds.\n",
                            "content": "These are the vehicle's CAM_FRONT historical images: 2.0s ago<image>\n1.5s ago<image>\n1.0s ago<image>\n0.5s ago<image>\n"+"These are the vehicle's current frame six-view images: CAM_FRONT:<image>\nCAM_FRONT_LEFT:<image>\nCAM_FRONT_RIGHT:<image>\nCAM_BACK:<image>\nCAM_BACK_LEFT:<image>\nCAM_BACK_RIGHT:<image>\n"+"Mission Goal:"+cmd_str+"\n"+"Historical trajectory:" + his_traj +", "+speed_content+". Based on the provided particulars, please generate BEV image 0.5s from now and plan waypoints (0.5s intervals) for the next 2 seconds.\n",
                            # "content": "These are the vehicle's CAM_FRONT historical images: 2.0s ago<image>\n1.5s ago<image>\n1.0s ago<image>\n0.5s ago<image>\n"+"These are the vehicle's current frame six-view images: CAM_FRONT:<image>\nCAM_FRONT_LEFT:<image>\nCAM_FRONT_RIGHT:<image>\nCAM_BACK:<image>\nCAM_BACK_LEFT:<image>\nCAM_BACK_RIGHT:<image>\n"+"Mission Goal:"+cmd_str+"\n"+"Historical trajectory:" + his_traj +", "+speed_content+". Based on the provided particulars, please generate BEV image at the current moment and plan waypoints (0.5s intervals) for the next 2 seconds.\n",
                            #"content": "These are the vehicle's CAM_FRONT historical images: 2.0s ago<image>\n1.5s ago<image>\n1.0s ago<image>\n0.5s ago<image>\n"+"These are the vehicle's current frame six-view images: CAM_FRONT:<image>\nCAM_FRONT_LEFT:<image>\nCAM_FRONT_RIGHT:<image>\nCAM_BACK:<image>\nCAM_BACK_LEFT:<image>\nCAM_BACK_RIGHT:<image>\n"+"Mission Goal:"+cmd_str+"\n"+"Historical trajectory: " + his_traj +"Based on the provided particulars, please plan waypoints (0.5s intervals) for the next 3 seconds.\n",
                            # "content": prompt+"These are the vehicle's CAM_FRONT images from the past 2.0s <image>\nImage from the past 1.5s<image>\nImage from the past 1.0s<image>\nImage from the past 0.5s<image>\n"+"Here are current six images from the car: CAM_FRONT:<image>\nCAM_FRONT_LEFT:<image>\nCAM_FRONT_RIGHT:<image>\nCAM_BACK:<image>\nCAM_BACK_LEFT:<image>\nCAM_BACK_RIGHT:<image>\n",
                            "role": "user"
                        },
                        {
                            # "content": "<think>"+think_content+" These are the visual tokens of CAM_FRONT image at the 0.5 second in the future. \n"+next_img_token+"</think>"+"<answer>"+"These are the future waypoints: "+ future_traj_str+"</answer>",
                            # "content": "<think>"+"These are the visual tokens of the BEV image at the current moment.\n"+next_img_token+"\nThese are the visual tokens of the BEV image 0.5 second from now.\n"+next_img_token1+"\nThese are the visual tokens of the BEV image 1.0 second from now.\n"+next_img_token2+"\nThese are the visual tokens of the BEV image 1.5 second from now.\n"+next_img_token3+"</think>"+"<answer>"+"These are the future pixel tokens:"+trajtoken+"\nThese are the future waypoints:"+ filtered_str+"</answer>",
                            # "content": "<think>"+"These are the visual tokens of the BEV image at the current moment.\n"+next_img_token+"</think>"+"<answer>"+"These are the future pixel tokens:"+trajtoken+"\nThese are the future waypoints:"+ filtered_str+"</answer>",
                            "content": "<think>"+think_content+"These are the visual tokens of the BEV image 0.5 second from now.\n"+next_img_token1+"</think>"+"<answer>"+"These are the future pixel tokens:"+trajtoken+"\nThese are the future waypoints:"+ filtered_str+"</answer>",
                            # "content": assitant_message,
                            # "content": "These are the future waypoints: "+ future_traj_str,
                            "role": "assistant"
                        }
                    ],
                    "images": final_path,
                    "id": id
                }

    train_messages.append(train_data_one)

print("#### Cost Summarization ####")

# with open(f"./LLaMA-Factory/data/{args.split}_cot_motion.json", "w") as f:
#     json.dump(train_messages, f, indent=4)
with open(f"./{args.split}_xingyun_v18_1028.json", "w") as f:
    json.dump(train_messages, f, ensure_ascii=False, indent=4)

