# Setup
I currently just use the motionagent virtual environment.
So create it from the motionagent folder and activate it before running any code:
```bash
    conda activate motionagent
```

# Generate motion from text
To generate motion from text, run the following command:
```bash
    python3 src/inference.py "a person walks slowly hunched over to the right."
```
You can set the parameters for generation and the output path in `src/config.py`.

# Train the model
To train the model, run the following command:
```bash
    python3 src/train.py
```
You can set the training parameters in `src/config.py`.