# A Multilayer Perceptron to generate continuous motion
This repo contains an MLP to generate motion from a previous start pose.

# 1. Environment Setup
```bash
conda create -n motionmlp python=3.10
conda activate motionmlp
pip install -r requirements.txt
```

# 2. Training
To start the training, run:
```bash
python3 src/train.py
```
The trained weights are stored in checkpoints/

# 3. Inference
To generate a motion with a random starting pose from the HumanML3D Dataset, run: 
```bash
python3 src/inference.py
```
You can use the pretrained weights or the ones you trained. You can set the path in `config.py`.
The number of generated frames can also be set in `config.py`.


