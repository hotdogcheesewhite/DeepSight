# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import oss2
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from typing import TYPE_CHECKING, Optional

from llamafactory.hparams import get_infer_args, get_ray_args, get_train_args, read_args
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, RoadCollector, ADCollector, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.misc import calculate_tps
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
# from data_tools.visual_train_data import visual_data


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from llamafactory.hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_eval(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    result_path: str = None
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module, preprocess_func = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, False)
    print('model dtype', model.dtype)
    model.eval()

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction
    # from pudb import set_trace; set_trace()
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    road_collator = ADCollector(preprocess_func, data_collator, augment=True, finetune=False)

    dataset = dataset_module["train_dataset"]
    k = min(100, len(dataset))
    random_indexs = random.sample(range(len(dataset)), k=k)
    # random_indexs = range(len(dataset))
    for i in tqdm(random_indexs):
        sample = dataset[i]
        batch_sample = road_collator([sample])
        sample_to_device(batch_sample, model.device)
        # pixel_values: Optional[torch.Tensor] = None,
        # pixel_values_videos: Optional[torch.FloatTensor] = None,
        # pixel_values_vectors: Optional[torch.FloatTensor] = None,
        # timestep: Optional[torch.Tensor] = None,
        # image_grid_thw: Optional[torch.LongTensor] = None,
        # video_grid_thw: Optional[torch.LongTensor] = None,
        with torch.no_grad():
            target_images, predict_images, losses = model.generate_img(
                pixel_values = batch_sample["pixel_values"],
                pixel_values_bevs = batch_sample["pixel_values_bevs"],
                timestep = batch_sample["timestep"],
                commands = batch_sample["commands"],
                image_grid_thw = batch_sample["image_grid_thw"],
            )
            all_predict_images, all_target_images = [], []
            for predict_image, target_image in zip(predict_images, target_images):
                # target image
                target_image = target_image.to(torch.float32)
                target_image = road_collator.vae_image_processor.postprocess(target_image, output_type="pil")
                all_target_images.append(target_image[0])
                # predict image
                predict_image = predict_image.to(torch.float32)
                predict_image = road_collator.vae_image_processor.postprocess(predict_image, output_type="pil")
                all_predict_images.append(predict_image[0])

        # 可视化查看
        visual_for_eval(all_target_images, all_predict_images, losses, sample, os.path.join(result_path, 'visual'))


def sample_to_device(batch_sample, device):
    for key in batch_sample:
        # for sub_key in batch_sample[key]:
        if type(batch_sample[key]) == torch.Tensor:
            cur_tensor = batch_sample[key]
            if cur_tensor.dtype == torch.float32:
                cur_tensor = cur_tensor.to(torch.bfloat16)
            # print(key, sub_key, cur_tensor.dtype)
            # print(cur_tensor.dtype)
            batch_sample[key] = cur_tensor.to(device)


def visual_for_eval(all_target_images, all_predict_images, losses, sample, visual_path):
    dst_h, dst_w = 364, 644
    bev_h, bev_w = 256, 256
    # 创建画布：
    canvas = np.zeros((dst_h * 3 + bev_h * 2, dst_w * 4, 3), dtype=np.uint8)
    
    
    # print(sample)
    img_file = sample['_images'][-1]
    clip_name = img_file.split('/')[-4]
    image_name = os.path.basename(img_file)
    # 绘制历史帧图像
    for i in range(4):
        img = cv2.imread(sample['_images'][i])
        img = cv2.resize(img, (dst_w, dst_h))
        canvas[:dst_h, dst_w * i:dst_w * (i + 1), :] = img
    # 绘制环视图
    indexs = [5, 4, 6, 8, 7, 9]
    for i, index in enumerate(indexs):
        img = cv2.imread(sample['_images'][index])
        img = cv2.resize(img, (dst_w, dst_h))
        h_i, w_i = i // 3 + 1, i % 3
        canvas[dst_h * h_i:dst_h * (h_i + 1), dst_w * w_i:dst_w * (w_i + 1), :] = img
    for index in range(len(all_predict_images)):
        # 绘制 BEV 图像
        target_img = all_target_images[index]
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        canvas[dst_h * 3 : dst_h * 3 + bev_h, bev_w * index : bev_w * (index + 1), :] = target_img
        # 绘制预测 BEV 图像
        predict_img = all_predict_images[index]
        predict_img = cv2.cvtColor(np.array(predict_img), cv2.COLOR_RGB2BGR)
        canvas[dst_h * 3 + bev_h : dst_h * 3 + bev_h * 2, bev_w * index : bev_w * (index + 1), :] = predict_img
        # 绘制损失
        loss = losses[index]
        cv2.putText(canvas, f"loss: {loss:.6f}", (bev_w * index + 20, dst_h * 3 + bev_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(os.path.join(visual_path, '_'.join([clip_name, image_name])), canvas)


if __name__ == "__main__":

    result_path = "debug/visual_for_bev_ex2_s13k_test"
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(f"{result_path}/visual", exist_ok=True)

    args = read_args()
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    run_eval(model_args, data_args, training_args, finetuning_args, result_path=result_path)