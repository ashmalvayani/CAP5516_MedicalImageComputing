<img width="704" alt="image" src="https://github.com/user-attachments/assets/0b5f117b-efaa-44a8-ae5b-2b73102c952c" />## Installation:

1. Run the following commands to install the environment for installing the code.

```shell
# Create the environment named 'assignment'
conda create -n assignment python=3.10 -y

# Activate the environment
conda activate assignment

# Install the packages from the requirements.txt file in the repository
cd finetune-SAM
pip install -r requirements.txt
```

## Downloading the NuInsSeg Dataset
2. Download the dataset

```shell
# I have downloaded the dataset from the link given in the assignment:
Download Link: Website (https://zenodo.org/records/10518968)
# or
Download Link: Kaagle (https://www.kaggle.com/datasets/ipateam/nuinsseg)

# From the above Link, download the NuInsSeg.zip, extract it, and transfer it to the cluster and place it under
datasets/
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
cd finetune-SAM
python k_fold.py
```

Now, since the required format for fine-tuning the dataset is train.csv and val.csv in the following format
> NuInsSeg/tissue_images/<img.png>, NuInsSeg/binary_masks/<img.png>

For this, run
```Shell
cd datasets/NuInsSeg/a_full_set_5fold/
python dataset_filter.py

# We will be using the files placed under datasets/NuInsSeg/a_full_set_5fold/final_files for our training and validation.
```


## Training
4. Run the following code for different training folds in the jobs folder:

```shell
# I am running the code by setting up the code using slurm scripts, to run the code, run the following command:

cd jobs
conda activate assignment

sbatch job0.slurm
sbatch job1.slurm
sbatch job2.slurm
sbatch job3.slurm
sbatch job4.slurm

# the logs will be saved in "outs/" folder.
```

## Prediction
5. Run the following code for evaluation by making following changes in the val_singlegpu_demo.sh:
Only change the below argument

> val_img_list="datasets/NuInsSeg/a_full_set_5fold/final_files/val_fold_{i}.csv"
> dir_checkpoint=Output_Directory_UsedIn_Training_Weights

```shell
# I am running the code by setting up the code using bash script, to run the code, run the following command:

conda activate assignment
bash val_singlegpu_demo.sh

# the outputs will be saved in the "NuInsSeg_Testing/" folder.
```
