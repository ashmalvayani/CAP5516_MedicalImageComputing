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

## Downloading the Brain-Tumor Segmentation Dataset
2. Download the dataset

```shell
# I have downloaded the dataset from the Google Drive from the given link:
Download Link: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2

# From the above Google Drive Link, download the Task01_BrainTumour.tar, extract it, and transfer it to the cluster and place it under
TransBTS/data/
```

## Dataset Processing
2. Since we have the original files in nii.gz format, first convert these files to the pickle files using the following command:
```Shell
cd TransBTS/data
python preprocess.py

# A few paths are hardcoded in the preprocess.py, you can change it to run it.
```

Next, we want to split the dataset into 5 random folds to compare the performance on each fold. We train on the training subset and test on the remainder of the files.

```Shell
# Change the path of the dataset in the file below and then run the following:
python k_fold.py
```


## Training
4. Run the following codes for different training folds in the jobs folder:

```shell
# I am running the code by setting up the code using slurm scripts, to run the code, run the following command:

cd jobs
conda activate assignment

sbatch train1.slurm
sbatch train2.slurm
sbatch train3.slurm
sbatch train4.slurm
sbatch train5.slurm

# the logs will be saved in "outs/" folder.
```

## Prediction
5. Run the following codes for different evaluation folds in the jobs folder:

```shell
# I am running the code by setting up the code using slurm scripts, to run the code, run the following command:

cd jobs
conda activate assignment

sbatch test1.slurm
sbatch test2.slurm
sbatch test3.slurm
sbatch test4.slurm
sbatch test5.slurm

# the logs will be saved in "pred_outs/" folder.
```
