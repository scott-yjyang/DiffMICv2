# DiffMIC-v2
<div align="center">
<br>
<h3>[IEEE TMI] DiffMIC-v2: Medical Image Classification via Improved Diffusion Network</h3>

<p align="center">
  <a href="[https://arxiv.org/abs/2506.02327](https://ieeexplore.ieee.org/abstract/document/10843287)"><img src="https://img.shields.io/badge/Paper-<COLOR>.svg" alt="Paper"></a>
  <a href="https://github.com/scott-yjyang/DiffMICv2"><img src="https://img.shields.io/badge/Dataset-yellow.svg" alt="Dataset"></a>
 <p align="center">
  
</div>

## News
- 24-01-26. This project is still quickly updating 🌝. Check TODO list to see what will be released next.
- 25-06-25. ❗❗Update on Code. Welcome to taste.😄
- 25-05. The paper has been released on arXiv.



## Abstract

Recently, Denoising Diffusion Models have achieved outstanding success in generative image modeling and attracted significant attention in the computer vision community. Although a substantial amount of diffusion-based research has focused on generative tasks, few studies apply diffusion models to medical diagnosis. 
In this paper, we propose a diffusion-based network (named DiffMIC-v2) to address general medical image classification by eliminating unexpected noise and perturbations in image representations. 
To achieve this goal, we first devise an improved dual-conditional guidance strategy that conditions each diffusion step with multiple granularities to enhance step-wise regional attention. 
Furthermore, we design a novel Heterologous diffusion process that achieves efficient visual representation learning in the latent space. 
We evaluate the effectiveness of our DiffMIC-v2 on four medical classification tasks with different image modalities, including thoracic diseases classification on chest X-ray, placental maturity grading on ultrasound images, skin lesion classification using dermatoscopic images, and diabetic retinopathy grading using fundus images. 
Experimental results demonstrate that our DiffMIC-v2 outperforms state-of-the-art methods by a significant margin, which indicates the universality and effectiveness of the proposed model on multi-class and multi-label classification tasks. 
DiffMIC-v2 can use fewer iterations than our previous DiffMIC to obtain accurate estimations, and also achieves greater runtime efficiency with superior results. 

<img width="800" height="600" src="https://github.com/scott-yjyang/DiffMICv2/blob/main/assets/framework.png">


## Environment Setup
### Clone this repository and navigate to the root directory of the project.

```bash
git clone https://github.com/scott-yjyang/DiffMICv2.git

cd DiffMICv2
```

### Install basic package

```bash
conda env create -f environment.yml
```


### Clone EfficientSAM

```bash
git clone https://github.com/yformer/EfficientSAM.git

```

## Datasets
Please refer to [DiffMIC](https://github.com/scott-yjyang/DiffMIC) for some details.




### TODO LIST

- [x] Release training scripts
- [x] Release evaluation
- [ ] Release Ultrasound dataset




## Acknowledgement

Code is developed based on [DiffMIC](https://github.com/scott-yjyang/DiffMIC), [EfficientSAM](https://github.com/yformer/EfficientSAM).

## Cite
If you find it useful, please cite and star
~~~
@article{yang2024vivim,
  title={Vivim: a Video Vision Mamba for Medical Video Object Segmentation},
  author={Yang, Yijun and Xing, Zhaohu and Zhu, Lei},
  journal={arXiv preprint arXiv:2401.14168},
  year={2024}
}
~~~
