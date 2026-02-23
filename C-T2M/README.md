# A Continuous Approach for Text to Motion Generation
This repository contains code for a text-to-motion generation model that generates motion sequences with a pretrained LLM while keeping the motion representation continuous in the latent space.

## Setup
First, cd into the project directory.
```bash
cd C-T2M
```
### Environment
Create and activate the conda environment.
```bash
conda create -n c-t2m python=3.10
conda activate c-t2m
pip install -r requirements.txt
```

### Download the HumanML3D dataset
You can download the HumanML3D dataset from [here](https://github.com/EricGuo5513/HumanML3D).
The file directory should look like this:
```
./dataset/HumanML3D/
├── new_joint_vecs/
├── texts/
├── Mean.npy
├── Std.npy
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```

### Download glove for evaluation
```bash
bash prepare/download_glove.sh
```

### Download evaluation models
For the evaluation, you need to download the pretrained models for FID, Diversity, Matching, and R-Precision.
You can download them from [here]. Unzip and place them under the checkpoint directory. It should look like this:
```
./checkpoints/t2m/
./checkpoints/t2m/Comp_v6_KLD01/           # Text-to-motion generation model
./checkpoints/t2m/Decomp_SP001_SM001_H512/ # Motion autoencoder (not needed here)
./checkpoints/t2m/length_est_bigru/        # Text-to-length sampling model (not needed here)
./checkpoints/t2m/text_mot_match/          # Motion & Text feature extractors for evaluation
```


## Generate motion from text
To generate motion from text, set the `INFERENCE_MODEL_PATH` in `src/config.py`.
Then run the following command:
```bash
python3 src/inference.py "a person walks slowly hunched over to the right."
```
You can set the parameters for generation and the output path in `src/config.py`.
The output will be a gif of the generated motion saved to the specified output path.

## Train the model
To train the model, run the following command:
```bash
python3 src/train.py
```
You can set the training parameters and the `CHECKPOINT_DIR` in `src/config.py`.

## Evaluation
To evaluate the model (FID, Diversity, Matching, R-Precision), set the model parameters in `src/config.py`.  
The `INFERENCE_MODEL_PATH` parameter in `src/config.py` should point to the trained model you want to evaluate.  
Then run the following command:
```bash
python3 src/evaluate.py
```