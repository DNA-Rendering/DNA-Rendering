# [ICCV2023] DNA-Rendering
[![arXiv](https://img.shields.io/badge/arXiv-2307.10173-b31b1b.svg)](https://arxiv.org/abs/2307.10173) <a href="https://dna-rendering.github.io/">
<img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a> 
<a href="https://youtu.be/xlhfvxvu7nc"><img alt="Demo" src="https://img.shields.io/badge/-Demo-ea3323?logo=youtube"></a> 

This is the official Benchmark PyTorch implementation of the paper *"[DNA-Rendering: A Diverse Neural Actor Repository for High-Fidelity Human-centric Rendering]()"*.



![renbody-teaser](https://github.com/DNA-Rendering/DNA-Rendering/assets/136057575/e64b8ca2-2490-46e7-a97e-a7bf05a0e34b)


> 
>
> **Abstract:** *Realistic human-centric rendering plays a key role in both computer vision and computer graphics. Rapid progress has been made in the algorithm aspect over the years, yet existing human-centric rendering datasets and benchmarks are rather impoverished in terms of diversity (e.g., outfit's fabric/material, body's interaction with objects, and motion sequences), which are crucial for rendering effect. Researchers are usually constrained to explore and evaluate a small set of rendering problems on current datasets, while real-world applications require methods to be robust across different scenarios. In this work, we present DNA-Rendering, a large-scale, high-fidelity repository of human performance data for neural actor rendering.
DNA-Rendering presents several alluring attributes. First, our dataset contains over 1500 human subjects, 5000 motion sequences, and 67.5M frames' data volume. Upon the massive collections, we provide human subjects with grand categories of pose actions, body shapes, clothing, accessories, hairdos, and object intersection,  which ranges the geometry and appearance variances from everyday life to professional occasions. Second, we provide rich assets for each subject -- 2D/3Dhuman body keypoints, foreground masks,  smplx models, cloth/accessory materials, multi-view images and videos. These assets boost the current method's accuracy on downstream rendering tasks. Third, we construct a professional multi-view system to capture data, which contains 60 synchronous cameras with max 4096 x 3000 resolution, 15fps speed, and stern camera calibration steps, ensuring high-quality resources for task training and evaluation.
Along with the dataset, we provide a large-scale and quantitative benchmark in full-scale, with multiple tasks to evaluate the existing progress of novel view synthesis, novel pose animation synthesis, and novel identity rendering methods. In this manuscript, we describe our DNA-Rendering effort as a revealing of new observations, challenges, and future directions to human-centric rendering. The dataset and code for data processing and benchmarking are publicly available at https://dna-rendering.github.io/ .* <br>

## Updates
- 2024.06.25: 🍹🍹🍹**A-pose data** of shared motion sequences🍹🍹🍹 is released!
- 2024.05.24: 🍮🍮🍮Data Part5🍮🍮🍮 is released! 
- 2024.05.13: 🏗️ We support directly using DNA-Rendering data in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) now. Please refer to (https://github.com/DNA-Rendering/DNA-Rendering/issues/12) for the instruction!
- 2024.05.11: 🍮🍮🍮Data Part4🍮🍮🍮 is released! 
- 2024.05.10: 🍹🍹🍹Data metioned in [4K4D](https://zju3dv.github.io/4k4d/) 🍹🍹🍹 is fully released!
- 2024.04.29: 🍮🍮🍮Data Part3🍮🍮🍮 is released! 
- 2024.03.08: 🍰🍰🍰Data metioned in [4K4D](https://zju3dv.github.io/4k4d/) 🍰🍰🍰 is released!
  > It is **NOT** part of the official DNA-Rendering dataset, we released for fair comparison in the research community.
- 2023.10.28: :fire::fire::fire:**[Data Part2](https://dna-rendering.github.io/inner-download.html) is fully released!**:fire::fire::fire: 
- 2023.09.28: 🍮🍮🍮Data Part2🍮🍮🍮 is released! 
- 2023.08.31: :fire::fire::fire:**[Data Part1](https://dna-rendering.github.io/inner-download.html) is fully released!**:fire::fire::fire: Except the raw RGB data and annotations, we also provide additional depth data!
- 2023.08.01: 🍮🍮🍮Data Part1🍮🍮🍮 is released! 
- 2023.07.20: :fire::fire::fire:**The [technical report](https://arxiv.org/abs/2307.10173) is released!**:fire::fire::fire:
- 2023.07.14: Our paper has been accepted by ICCV 2023!
- 2023.07.01: Technical report, data and code will be released soon. Please stay tuned!
- 2023.07.01: The [demo video](https://www.youtube.com/watch?v=C5mtexVS3DU) is uploaded. Check it out for an overview of this project!
- 2023.07.01: The [project page](https://dna-rendering.github.io/) is created.


## Contents
1. [Features](#features)
2. [Data Download](#Data-Download)
3. [Benchmark & Model Zoo](#Benchmark-&-Model-Zoo)
4. [Usage](#Usage)
5. [Related Works](#Related-Works)
6. [Citation](#citation)
<!--6. [Acknowlegement](#Acknowlegement)-->


## Features
* Scales: To our knowledge, our dataset far surpasses similar ones in terms of the number of actors, costumes, actions, clarity, and overall data volume.
  
https://github.com/DNA-Rendering/DNA-Rendering/assets/136057575/a6b3d561-38a1-4323-8c9a-ab4fa3e8f227
* Diversity: Our dataset covers a diverse range of scenarios, including everyday and special performances. It includes large variation of clothing and action types, with sufficient difficulty levels in clothing textures and motion complexity. This diversity and difficulty make it suitable for a variety of downstream research tasks.


https://github.com/DNA-Rendering/DNA-Rendering/assets/136057575/35712e04-c8e6-4158-97de-7b9763a08069
* High-quality Annotations: Our dataset comes with off-the-shelf high-precision annotation, including 2D/3D human body keypoints, foreground masks, and SMPL-X models. We have specifically optimized our annotations for 3D human body scenarios, resulting in high-quality annotations.

https://github.com/DNA-Rendering/DNA-Rendering/assets/136057575/643d41f5-ab74-420b-af8d-e60a8cf5732e
* Benchmarks: We have provided the results of various state-of-the-art methods of rendering and animation on our dataset.
![Benchmark](https://github.com/DNA-Rendering/DNA-Rendering/assets/136057575/f4bd098a-48c9-4645-b65b-78e8760b8b5a)

## Data Download
The dataset will be released soon.

# Benchmark & Model Zoo

Coming soon! We provide for each benchmark the pretrained model, code for training & evaluation reimplementation, and dataset for training.

| Benchmark                          | Aspect                           | Pretrained Model                                                | Reimplementation                     | Dataset                          |
| -------------------------------    | -------------------------------  | ------------------------------------------------------------ | ---------------- | -------------------------------------------- |
| instant-ngp    | NovelView             |  | | |
| NeuS           | NovelView             |  | | |
| Neural Volumes | NovelView/NovelPose   |  | | |
| A-NeRF         | NovelView/NovelPose   |  | | |
| Neural Body    | NovelView/NovelPose   |  | | |
| Animatable Nerf| NovelView/NovelPose   |  | | |
| HumanNeRF      | NovelView/NovelPose   |  | | |
| IBRNet         | NovelID/CrossData     |  | | |
| Pixel          | NovelID/CrossData     |  | | |
| Vision         | NovelID/CrossData     |  | | |
| Neural Human Performance   | NovelID   |  | | |
| KeyPointNerf   | NovelID               |  | | |

## Usage
The code will be released soon!

## TODO List

- [ ] Release Code and pretrained model
- [ ] Release Dataset
- [x] Technical Report
- [x] Project page


## Related Works
## Citation

```bibtex
@article{2023dnarendering,
      title={DNA-Rendering: A Diverse Neural Actor Repository for High-Fidelity Human-centric Rendering}, 
      author={Wei Cheng and Ruixiang Chen and Wanqi Yin and Siming Fan and Keyu Chen and Honglin He and Huiwen Luo and Zhongang Cai and Jingbo Wang and Yang Gao and Zhengming Yu and Zhengyu Lin and Daxuan Ren and Lei Yang and Ziwei Liu and Chen Change Loy and Chen Qian and Wayne Wu and Dahua Lin and Bo Dai and Kwan-Yee Lin},
      journal   = {arXiv preprint},
      volume    = {arXiv:2307.10173},
      year    = {2023}
}
```
<!-- ## Acknowlegement -->

