<div align="center">

  <h1 align="center">DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving</h1>

<h3 align="center"><strong>🎉🎉ICML 2026 🎉🎉</strong></h3>

</div>

<div align="center">
<img width="800" alt="image" src="assets/fig1.png">
<p>DeepSight achieves leading performance on most of metrics compared with E2E methods.</p>
</div>

## 📖 Abstract

End-to-end autonomous driving systems are increasingly integrating Vision-Language Model (VLM) architectures, incorporating text reasoning or visual reasoning to enhance the robustness and accuracy of driving decisions.
However, the reasoning mechanisms employed in most methods are direct adaptations from general domains, lacking in-depth exploration tailored to autonomous driving scenarios, particularly within visual reasoning modules. In this paper, we propose a driving world model that performs parallel prediction of latent semantic features for consecutive future frames in the bird’s-eye-view (BEV) space, thereby enabling long-horizon modeling of future world states. We also introduce an efficient and adaptive text reasoning mechanism that utilizes additional social knowledge and reasoning capabilities to further improve driving performance in challenging long-tail scenarios. We present a novel, efficient, and effective approach that achieves state-of-the-art (SOTA) results on the closed-loop Bench2drive benchmark.

## 🚀 Pipeline

<div align="center">
<img width="800" alt="image" src="assets/figmethod.png">
<p>The pipeline of our method, a holistic training and inference framework for closed-loop driving. It consists of two main modules:
(a) Long-term driving-world model, for aligning DINOv3 features extracted from future multi-frame RGB images in the BEV space
during training. (b) An adaptive CoT module for integrating external knowledge to enhance reasoning and decision-making in long-tail
cases</p>
</div>

## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="assets/figvis.png">
<p>Qualitative results of DeepSight on the Bench2Drive closed-loop evaluation set.</p>
</div>
# DeepSight

DeepSight is an autonomous driving perception and reasoning framework built on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), with full-pipeline customization for data processing, model training, and inference evaluation in autonomous driving scenarios.

## Quick Start

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/hotdogcheesewhite/DeepSight.git
cd DeepSight

# Create virtual environment
conda create -n deepsight python=3.10 -y && conda activate deepsight

# Install PyTorch (recommended 2.6.0 or other compatible version)
# torch == 2.6.0

# Install dependencies
pip install -r requirements.txt
# pip install -e .
```

## Overview

This project extends LLaMA-Factory with the following features:

- **BEV (Bird's-Eye-View) data processing and visualization pipeline**
- **VLM (Vision-Language Model) training data construction**
- **DINOv3 feature extraction and BEV Query supervision**
- **Open-loop and closed-loop evaluation based on Bench2Drive**

## Directory Structure

```
deepsight/
├── configs/                                  # Training configuration files (YAML)
├── data/                                     # Dataset processing utilities
├── src/                                      # Core source code
│   ├── tools/                                # Data & evaluation utility scripts
│   │   ├── crop_bev_for_bench2drive.py       # BEV image cropping
│   │   ├── visual_for_bev.py                 # BEV visualization
│   │   ├── create_date_set.py                # VLM training data construction
│   │   ├── eval_and_visual.py                # Inference visualization & open-loop eval
│   │   └── merge_model_weight.py             # Model weight merging (for vLLM)
│   ├── transformers/src/transformers/        # Modified transformers
│   │   └── models/qwen2_5_vl/modeling_*.py    # Qwen2.5-VL model (with DINOv3)
│   ├── llamafactory/data/ad_collator.py      # Data collator (removes token CE loss)
│   ├── infer_for_debug.py                    # Original transformers inference
│   └── infer_with_vllm.py                    # vLLM inference
├── bench2drive/                              # Bench2Drive evaluation framework
│   └── leaderboard/scripts/
│       └── run_evaluation_qwen.sh            # Closed-loop evaluation script
├── nebula.sh                                 # Nebula cluster training script
└── requirements.txt                          # Training environment dependencies
```

---

## 1. Data Preparation

### 1.1 Create BEV Maps

**Script:** [src/tools/crop_bev_for_bench2drive.py](src/tools/crop_bev_for_bench2drive.py)

Crops BEV (Bird's-Eye-View) images from Bench2Drive data. Each BEV includes **5 fixed-resolution future motion images** for vehicle trajectory prediction.

**Notes:**
- Highly sensitive to weather conditions — can be mitigated by lowering the BEV height
- Perspective distortion from nearby tall buildings

**Visualization check:** [src/tools/visual_for_bev.py](src/tools/visual_for_bev.py)

Focus on verifying data quality in **turning/cornering scenes**.

### 1.2 Create VLM Training Data

**Script:** [bench2drive/dataprocess/targetpointgen.py](bench2drive/dataprocess/targetpointgen.py)

Converts raw data into conversational format for training. Input requires annotation files.

### 1.3 (Optional) Manually Construct CoT Annotation Content

**Script:** [src/tools/create_date_set_target_need_to_cot.py](src/tools/create_date_set_target_need_to_cot.py)

Replace textprompt with desired prompts to generate data that needs Qwen-3VL annotation.

### 1.4 (Optional) Call API to Generate Annotation Data

**Script:** [bench2drive/dataprocess/jsonopenai.py](bench2drive/dataprocess/jsonopenai.py)

Update the OpenAI API key and call the Qwen3VL model to perform annotation.
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>
---

## 2. Model Training

### Training Entry Point

1. Add the corresponding dataset information in `deepsight/data/dataset_info.json`. Refer to the [LLaMA-Factory official documentation](https://llamafactory.readthedocs.io/) to organize the dataset and modify the path to the previously generated JSONL file.

2. Execute the following command:

```bash
bash nebula.sh
```

The training entry point is `src/train.py`, with hyperparameters defined in YAML configuration files under the `configs/` directory.

### Loss Design

[src/llamafactory/data/ad_collator.py](src/llamafactory/data/ad_collator.py)

Designs the training method.
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>
---

## 3. Model Inference

### 3.1 Inference with Original Transformers

**Script:** [src/infer_for_debug.py](src/infer_for_debug.py)

Uses the modified original transformers for inference (includes DINOv3 and other modules).

- **Visualization:** [src/tools/eval_and_visual.py](src/tools/eval_and_visual.py)
- **Open-loop evaluation:** [src/tools/eval_and_visual.py](src/tools/eval_and_visual.py)

### 3.2 Inference with vLLM

**Script:** [src/infer_with_vllm.py](src/infer_with_vllm.py)

vLLM uses an internal implementation of transformers that **does not include DINOv3** or other custom modules. Model weights must be merged before inference:

- **Merge script:** [src/tools/merge_model_weight.py](src/tools/merge_model_weight.py)
- Fixed target path: `/mnt/nas-data-1/wuchangjie.wcj/work/bev_ex3_v3_fulldata_resume/Qwen2.5-VL-3B-Instruct`
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>
---

## 4. Closed-Loop Evaluation

### Step 1: Install CARLA

**Note:** CARLA can only be used by non-root users.

Install CARLA (0.9.16 has Python 3.10 support, and Python 3.10 is required for large models):

```bash
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz
tar -xvf CARLA_0.9.16.tar.gz
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.16.tar.gz
bash ImportAssets.sh
```

After extraction, execute:

```bash
/mnt/nas-data-1/zhanglingjun.zlj1/carla/carla0916/ImportAssets.sh
```

This will extract the additional maps.

Then create a Python 3.10 CARLA environment:

```bash
conda activate /mnt/nas-data-1/zhanglingjun.zlj_env/envs/carla/
pip install carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
```

Start CARLA:

```bash
./CarlaUE4.sh -RenderOffScreen -nosound -fps=10 -carla-rpc-port=2000
```

`-RenderOffScreen` indicates headless mode (no GUI).

**Check if it's working:**

The key is that `vulkaninfo | grep "GPU id"` can detect the physical machine. If it can detect, then CARLA startup should be fine.

**If it cannot start normally, consider:**

```bash
cd /etc/vulkan
```

Check if there are any JSON files under `icd.d/`.

Under `icd.d/`:

```bash
sudo touch nvidia_icd.json
```

Write:

```json
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0",
        "api_version" : "1.3.277"
    }
}
```

Under `implicit_layer.d/`:

```bash
sudo touch nvidia_layers.json
```

Write:

```json
{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libEGL_nvidia.so.0",
        "api_version" : "1.3.277",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }
}
```

```bash
./CarlaUE4.sh -RenderOffScreen -nosound -fps=10 -carla-rpc-port=2000 -graphicsadapter=5
```

### Step 2: Install Environment

```bash
cd bench2drive
conda create -n b2d python=3.10 -y && conda activate b2d
```

Choose torch based on different CUDA versions (vLLM 0.8.0 requires torch 2.6.0):

```bash
export PATH=YOUR_GCC_PATH/bin:$PATH
export CUDA_HOME=YOUR_CUDA_PATH/
```

Basically, `nvcc --version` should display normally, and `gcc` and `g++ --version` should display normally.

cd to bench2drive zoo folder:

```bash
pip install ninja packaging
pip install -v -e .
```

In the environment:

```
numba==0.61.2  # In order to speed up
numpy==1.26.4  # In order to adapt numba
```

You need to modify the content in bench2drive's requirements.

### Step 3: Install QwenVL Inference Environment

A reference inference environment example: `deepsight/example.txt`

**Script:** [bench2drive/leaderboard/scripts/run_evaluation_qwen.sh](bench2drive/leaderboard/scripts/run_evaluation_qwen.sh)

Runs the Bench2Drive evaluation leaderboard pipeline with the trained Qwen model. Requires the separate inference environment installed in `bench2drive/`.
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>
---

## Key Components

| Component | Description |
|-----------|-------------|
| `nebula.sh` | Training job submission script for Nebula cluster |
| `requirements.txt` | Training environment Python dependencies |
| `configs/` | YAML configuration files for different training runs |
| `bench2drive/` | Bench2Drive evaluation framework (separate inference environment) |
| `src/tools/` | Data processing, visualization, and evaluation utility scripts |
| `src/transformers/` | Modified transformers (with DINOv3 integration) |
| `src/llamafactory/data/ad_collator.py` | Autonomous driving data collator |

## License

This project is distributed under the terms described in the [LICENSE](LICENSE) file.

## 🗓️ TODO
- [ ] Release DeepSight reasoning code
- [ ] Release whole DeepSight code
- [ ] Release checkpoints
<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>

## 🙏 Acknowledgement
Our work is primarily based on the following codebases:[FSDrive](https://github.com/MIV-XJTU/FSDrive), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [MoVQGAN](https://github.com/ai-forever/MoVQGAN), [GPT-Driver](https://github.com/PointsCoder/GPT-Driver), [Agent-Driver](https://github.com/USC-GVL/Agent-Driver). We are sincerely grateful for their work.

<p align="right"><a href="#readme-top"><img src=https://img.shields.io/badge/back%20to%20top-red?style=flat
></a></p>
