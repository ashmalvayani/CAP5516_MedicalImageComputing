import pickle
import os
import numpy as np
import nibabel as nib
import json

modalities = ('flair', 't1ce', 't1', 't2')

# train
train_set = {
        'root': 'Task01_BrainTumour',
        'flist': 'all.txt',
        'has_label': True
        }

# test/validation data
valid_set = {
        'root': 'path to validation set',
        'flist': 'valid.txt',
        'has_label': False
        }

test_set = {
        'root': 'path to testing set',
        'flist': 'test.txt',
        'has_label': False
        }


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def process_i16(path, has_label=True):
    """ Save the original 3D MRI images with dtype=int16.
        Noted that no normalization is used! """
    label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')

    images = np.stack([
        np.array(nib_load(path + modal + '.nii.gz'), dtype='int16', order='C')
        for modal in modalities], -1)# [240,240,155]

    output = path + 'data_i16.pkl'

    with open(output, 'wb') as f:
        print(output)
        print(images.shape, type(images), label.shape, type(label))  # (240,240,155,4) , (240,240,155)
        pickle.dump((images, label), f)

    if not has_label:
        return


def process_f32b0(file_path, label_path, has_label=True):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    # if has_label:
    
    label = np.array(nib_load(label_path), dtype='uint8', order='C')
    images = np.array(nib_load(file_path), dtype='float32', order='C')
    
    # label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')
    # images = np.stack([np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C') for modal in modalities], -1)  # [240,240,155]

    print(file_path)
    file_name = file_path.split('/')[-1].replace('.nii.gz', '')

    output = f"/home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/data/pickle_files/{file_name}.pkl"
    mask = images.sum(-1) > 0
    for k in range(4):

        x = images[..., k]  #
        y = x[mask]

        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x

    with open(output, 'wb') as f:
        print(output)

        if has_label:
            pickle.dump((images, label), f)
        else:
            pickle.dump(images, f)

    if not has_label:
        return


def doit(file_paths, label_paths):
    for file_path, label_path in zip(file_paths, label_paths):

        # print(file_path, label_path)
        process_f32b0(file_path, label_path)


def preprocess_paths():
    dataset_file = "/home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/data/Task01_BrainTumour/dataset.json"

    with open(dataset_file, "r") as file:
        data = json.load(file)

    file_names = data["training"]

    final_files = []
    label_files = []
    for file_name in file_names:
        file_path = file_name['image'].replace('./', '')
        file_path = os.path.join("/home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/data/Task01_BrainTumour", file_path)

        label_path = file_name['label'].replace('./', '')
        label_path = os.path.join("/home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/data/Task01_BrainTumour", label_path)
        # print(file_path, label_path, sep='\n')

        final_files.append(file_path)
        label_files.append(label_path)

    return final_files, label_files


if __name__ == '__main__':
    final_files, label_files = preprocess_paths()
    doit(final_files, label_files)

    # doit(train_set)
    # doit(valid_set)
    # doit(test_set)

