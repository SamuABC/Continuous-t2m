# Continuous Motion in Text-to-Motion Models
This repository contains two projects that aim to explore continuous motion representations in the latent space of text-to-motion generation models. Motion-MLP was used to get familiar with the topic and C-T2M is the text-to-motion model, which was trained to answer the following research question:

## Continuous Text-to-Motion (C-T2M)
Current work in human motion generation often utilizes Vector Quantized Variational Autoencoders to discretize continuous human motion into a fixed vocabulary of tokens, followed by next-token prediction using a pretrained Large Language Model (LLM). However, this approach introduces quantization errors that limit motion quality. This project introduces Continuous Text-to-Motion (C-T2M), a new autoregressive text-to-motion model that uses a continuous motion representation to avoid these quantization errors. C-T2M encodes motion with an MLP autoencoder and adapts a pretrained LLM backbone with Low-Rank Adaptation (LoRA) for motion generation. An ablation study was conducted to evaluate the influence of different training strategies, objective functions, and motion embeddings on the model’s performance. The best results are achieved with a scheduled sampling training strategy, a combination of position and velocity losses, and by training the motion autoencoder together with the LoRA parameters. The experimental results demonstrated that quantization errors are successfully eliminated with this approach. Still, the generated motions lack semantic alignment with the input prompts and need refinement in motion quality. The main reasons for these issues are identified as error accumulation within continuous autoregressive generation and the regression to the mean resulting from the mean squared error loss formulation. Future research should focus on hybrid models that combine autoregressive generation with diffusion-based approaches.

Information about the setup of C-T2M can be found in the [C-T2M README](C-T2M/README.md).

<img width="622" height="513" alt="grafik" src="https://github.com/user-attachments/assets/5a2b83b9-f72b-4238-a2ec-c3507668299b" />


## Motion-MLP
This is just a simple autoregressive MLP that generates random motion without any text input. It was implemented to get familiar with continuous motion representations.  
Setup information for this project can be found in the [Motion-MLP README](Motion-MLP/README.md).

## Dataset & Acknowledgments
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
