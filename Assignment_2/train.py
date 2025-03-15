import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle
from tqdm import tqdm

import torch
import torch.optim
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from models import criterions
from data.BraTS import BraTS
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch import nn

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='name of user', type=str)
parser.add_argument('--experiment', default='TransBTS', type=str)
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
parser.add_argument('--description', default='TransBTS, training on train.txt!', type=str)
# DataSet Information
parser.add_argument('--root', default='/home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/data/Task01_BrainTumour', type=str)
parser.add_argument('--train_dir', default='imagesTr', type=str)

parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--train_file', default='train.txt', type=str)
parser.add_argument('--dataset', default='brats', type=str)
parser.add_argument('--input_C', default=4, type=int)
parser.add_argument('--input_H', default=240, type=int)
parser.add_argument('--input_W', default=240, type=int)
parser.add_argument('--input_D', default=160, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--num_class', default=4, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
# parser.add_argument('--criterion', default='softmax_dice', type=str)
parser.add_argument('--criterion', default='softmax_dice2', type=str)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--end_epoch', default=500, type=int)
# parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--load', default=True, type=bool)

parser.add_argument('--k_fold', required=True, default='', type=int)
parser.add_argument('--checkpoint_dir', default='/home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/checkpoints', type=str)

parser.add_argument('--pickle_files', default='/home/ashmal/Courses/MedImgComputing/Assignment_2/TransBTS/data/pickle_files', type=str)

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")

    model.to(device)
    model.train()

    print("model loaded")
    # print(model.summary())

    pickle_files = args.pickle_files
    pickle_files_list = os.listdir(pickle_files)
    # pickle_files = [os.path.join(pickle_files, file) for file in pickle_files_list]

    train_files = []
    with open(args.train_file) as f:
        for line in f:
            name = line.strip()
            train_files.append(name)

    final_files = [file for file in pickle_files_list if file in train_files]
    pickle_files = [os.path.join(pickle_files, file) for file in final_files]


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    criterion = getattr(criterions, args.criterion)
    
    # train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)

    train_set = BraTS(pickle_files, train_root, args.mode)
    
    # train_set = BraTS(train_list, train_root, args.mode, pickle_files)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    print("Dataset Loaded.\n")
    
    writer = SummaryWriter()
    # checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment + args.date)
    checkpoint_dir = os.path.join(args.checkpoint_dir, f"fold_{args.k_fold}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.isfile(args.resume) and args.load:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        logging.info(f"Loaded checkpoint {args.resume} from epoch {args.start_epoch}")
    
    
    print("\n================ Started Training ===========\n")
    for epoch in tqdm(range(args.start_epoch, args.end_epoch)):
        print(f"\n================ EPOCH: {epoch}  ===========\n")
        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        # for i, (x, target) in tqdm(enumerate(train_loader)):
        for i, (x, target) in bar:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()

            # print("Training data x:shape: ", x.shape)
            # exit()
            output = model(x)
            loss, _, _, _ = criterion(output, target)
            loss.backward()
            optimizer.step()
            logging.info(f"Epoch: {epoch}, Iter: {i}, Loss: {loss.item():.5f}")

            bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))
        
    checkpoint_path = os.path.join(checkpoint_dir, f"fold_{args.k_fold}.pth")
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict()}, checkpoint_path)
    
    writer.close()
    torch.save({'epoch': args.end_epoch, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict()}, os.path.join(checkpoint_dir, f"fold_{args.k_fold}.pth"))

if __name__ == "__main__":
    main()
