# A Multilayer Perceptron to generate continuous motion
This repo contains an MLP to generate motion from a previous start pose. This start pose can be randomly sampled from the HumanML3D dataset.

## Environment Setup
```bash
conda create -n motionmlp python=3.10
conda activate motionmlp
pip install -r requirements.txt
```

# Download HumanML3D Dataset
Download the HumanML3D dataset and place it in `./data/HumanML3D`. More detailed information about how to download the dataset can be found [here](../T2M-MLP/README.md).

## Training
To start the training, run:
```bash
python3 src/train.py
```
The trained weights are stored in checkpoints/

## Inference
To generate a motion with a random starting pose from the HumanML3D Dataset, run: 
```bash
python3 src/inference.py
```
You can use the pretrained weights or the ones you trained. You can set the path in `config.py`.
The number of generated frames can also be set in `config.py`.


