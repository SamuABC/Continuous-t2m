# Continuous Motion in Text-to-Motion Models
This repository contains two projects that aim to explore continuous motion representations in the latent space of text-to-motion generation models.

## Motion-MLP
This is just a simple autoregressive MLP that generates random motion without any text input. It was implemented to get familiar with continuous motion representations.  
More information about this project can be found in the [Motion-MLP README](Motion-MLP/README.md).

## Continuous Text-to-Motion (C-T2M)
This is the main project that implements a text-to-motion generation model using a pretrained large language model (LLM) while keeping the motion representation continuous in the latent space.
More information about this project can be found in the [C-T2M README](C-T2M/README.md).

### Dataset & Acknowledgments
C-T2M was trained on the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset.
```bibtex
@InProceedings{Guo_2022_CVPR,
    author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
    title     = {Generating Diverse and Natural 3D Human Motions From Text},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5152-5161}
}
```
