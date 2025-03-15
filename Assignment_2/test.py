import argparse
import os
import time
import random
import numpy as np
import setproctitle

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader

from data.BraTS import BraTS
from predict import validate_softmax
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS


parser = argparse.ArgumentParser()

parser.add_argument('--user', default='name of user', type=str)
parser.add_argument('--root', default='path to testing set', type=str)
parser.add_argument('--valid_dir', default='Valid', type=str)
parser.add_argument('--valid_file', default='valid.txt', type=str)
parser.add_argument('--output_dir', default='/home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/inference/', type=str)
parser.add_argument('--submission', default='submission', type=str)
parser.add_argument('--visual', default='visualization', type=str)
parser.add_argument('--experiment', default='', type=str)
parser.add_argument('--test_date', default='', type=str)
parser.add_argument('--use_TTA', default=True, type=bool)
parser.add_argument('--post_process', default=True, type=bool)
parser.add_argument('--save_format', default='nii', choices=['npy', 'nii'], type=str)
parser.add_argument('--crop_H', default=128, type=int)
parser.add_argument('--crop_W', default=128, type=int)
parser.add_argument('--crop_D', default=128, type=int)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--model_name', default='TransBTS', type=str)
parser.add_argument('--num_class', default=4, type=int)
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--checkpoint_dir', default="/home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/checkpoints", type=str)
parser.add_argument('--k_fold', required=True, default=None, type=int)
parser.add_argument('--test_file', required=True, default='', type=str)
parser.add_argument('--pickle_files', default='/home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/data/pickle_files', type=str)

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    load_file = os.path.join(args.checkpoint_dir, f"fold_{args.k_fold}/fold_{args.k_fold}.pth")

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(load_file))
    else:
        print('There is no resume file to load!')


    pickle_files = args.pickle_files
    pickle_files_list = os.listdir(pickle_files)

    test_files = []
    with open(args.test_file) as f:
        for line in f:
            name = line.strip()
            test_files.append(name)

    final_files = [file for file in pickle_files_list if file in test_files]
    valid_files = [os.path.join(pickle_files, file) for file in final_files]

    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraTS(valid_files, valid_root, mode='test')

    print('Samples for valid = {}'.format(len(valid_set)))

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    submission = os.path.join(args.output_dir, args.submission, f"fold_{args.k_fold}")
    visual = os.path.join(args.output_dir, args.visual, f"fold_{args.k_fold}")

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    with torch.no_grad():
        validate_softmax(valid_loader=valid_loader,
                         model=model,
                         load_file=load_file,
                         multimodel=False,
                         savepath=submission,
                         visual=visual,
                         names=valid_set.names,
                         use_TTA=args.use_TTA,
                         save_format=args.save_format,
                         snapshot=True,
                         postprocess=True
                         )

    end_time = time.time()
    full_test_time = (end_time-start_time)/60
    average_time = full_test_time/len(valid_set)
    print('{:.2f} minutes!'.format(average_time))


if __name__ == '__main__':
    # config = opts()
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()


