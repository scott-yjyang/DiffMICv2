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
- 25-06-15. The paper is listed as IEEE TMI Popular Paper of May 2025.
- 25-01-15. The paper is accepted by IEEE Transactions on Medical Imaging.


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
conda create -n DiffMICv2 python=3.8
conda activate DiffMICv2
pip install -r requirements.txt
```


### Clone EfficientSAM

```bash
git clone https://github.com/yformer/EfficientSAM.git

```

## Run
### Train
```bash
python diffuser_trainer.py

(trainer.fit(model,ckpt_path=resume_checkpoint_path))
```

### Validate
```bash
python diffuser_trainer.py

(trainer.validate(model,ckpt_path=val_path))
```


## Datasets
Please refer to [DiffMIC](https://github.com/scott-yjyang/DiffMIC) for some details.




## TODO LIST

- [x] Release training scripts
- [x] Release evaluation
- [ ] Release Ultrasound dataset



## Acknowledgement

Code is developed based on [DiffMIC](https://github.com/scott-yjyang/DiffMIC), [EfficientSAM](https://github.com/yformer/EfficientSAM).

This project is under CC BY-NC 2.0. All Copyright © [Yijun Yang](https://yijun-yang.github.io/)

## Cite
If you find it useful, please cite and star
~~~
@article{yang2025diffmic,
  title={DiffMIC-v2: Medical Image Classification via Improved Diffusion Network},
  author={Yang, Yijun and Fu, Huazhu and Aviles-Rivero, Angelica I and Xing, Zhaohu and Zhu, Lei},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
~~~
