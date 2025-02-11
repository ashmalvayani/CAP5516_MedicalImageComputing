## Installation:

1. Run the following commands to install the environment for installing the code.

```shell
# Create the environment named 'assignment'
conda create -n assignment python=3.10 -y

# Activate the environment
conda activate assignment

# Install the packages from the requirements.txt file in the repository
pip install -r requirements.txt
```

## Downloading the Chest-Xray Dataset
2. Download the dataset

```shell
# I have downloaded the dataset from Kaggle. Sign up to create your account and set up your authentication API/token by following the steps from this website: https://www.kaggle.com/docs/api#authentication and login into the terminal using kagglehub library.

import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)
cp -r path data/
```

## Training
3. Run the code

```shell
# I am running the code by setting up the code using slurm scripts, to run the code, run the following command:

sbatch train.slurm
```

## Model Output weights:
You can load the model weights for both pre-trained and trained from scratch through the drive link below:
https://drive.google.com/drive/folders/1bTHw2QlLExUYPInQb6Gt9luk0EtkYPSz?usp=sharing

