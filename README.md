<div align="center">

  <h1 align="center">DeepSight: Long-Horizon World Modeling via Latent States Prediction for End-to-End Autonomous Driving</h1>



</div>

<div align="center">
<img width="800" alt="image" src="asserts/fig1.png">
<p>DeepSight achieves leading performance on most of metrics compared with E2E methods.</p>
</div>

## 📖 Abstract

End-to-end autonomous driving systems are increasingly integrating Vision-Language Model (VLM) architectures, incorporating text reasoning or visual reasoning to enhance the robustness and accuracy of driving decisions.
However, the reasoning mechanisms employed in most methods are direct adaptations from general domains, lacking in-depth exploration tailored to autonomous driving scenarios, particularly within visual reasoning modules. In this paper, we propose a driving world model that performs parallel prediction of latent semantic features for consecutive future frames in the bird’s-eye-view (BEV) space, thereby enabling long-horizon modeling of future world states. We also introduce an efficient and adaptive text reasoning mechanism that utilizes additional social knowledge and reasoning capabilities to further improve driving performance in challenging long-tail scenarios. We present a novel, efficient, and effective approach that achieves state-of-the-art (SOTA) results on the closed-loop Bench2drive benchmark.

## 🚀 Pipeline

<div align="center">
<img width="800" alt="image" src="asserts/figmethod.png">
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

## 🗓️ TODO
- [ ] Release DeepSight reasoning code
- [ ] Release whole DeepSight code
- [ ] Release checkpoints
