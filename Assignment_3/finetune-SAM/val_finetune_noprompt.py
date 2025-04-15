#from segment_anything import SamPredictor, sam_model_registry
from models.sam import SamPredictor, sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from skimage.measure import label
from models.sam_LoRa import LoRA_Sam
#Scientific computing 
import numpy as np
import os
#Pytorch packages
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import datasets
#Visulization
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
#Others
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from utils.dataset import Public_dataset
import torch.nn.functional as F
from torch.nn.functional import one_hot
from pathlib import Path
from tqdm import tqdm
from utils.losses import DiceLoss
from utils.dsc import dice_coeff, dice_coeff_multi_class
import cv2
import monai
from utils.utils import vis_image
import cfg
from argparse import Namespace
import json

def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from 
    over-penalisation similar to DICE2.
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true).astype(int))
    pred_id_list = list(np.unique(pred).astype(int))
    #print(len(pred_id_list))
    if len(pred_id_list) == 1:
        return 0

    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred = pred.squeeze(0)
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id.astype(int))
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    #print(aji_score)
    return aji_score


def get_fast_pq(true, pred, match_iou=0.5):
    """
    `match_iou` defines the IoU threshold used to determine valid matches between 
    ground truth (GT) instances `p` and predicted instances `g`. A pair (`p`, `g`) 
    is considered a match if IoU(p, g) > `match_iou`. Each GT instance can be 
    matched with only one predicted instance, and vice versa — ensuring a 1-to-1 
    mapping.

    - If `match_iou` < 0.5, Munkres (Hungarian) algorithm is used to compute the 
    maximum number of unique matches by solving a minimum-weight bipartite 
    matching problem.
    - If `match_iou` ≥ 0.5, any pair with IoU > `match_iou` is guaranteed to be 
    uniquely matched, and the number of such matches is already maximal.

    Note: For faster computation, instance IDs should be in a continuous sequence 
    (e.g., [1, 2, 3, 4]) instead of arbitrary values (e.g., [2, 3, 6, 10]). Use 
    `remap_label` beforehand to ensure this. The `by_size` parameter does not 
    influence the result.

    Returns:
        - [dq, sq, pq]: evaluation metrics
        - [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
        pairing details used in the evaluation
    """

    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true).astype(int))
    pred_id_list = list(np.unique(pred).astype(int))
    
    if len(pred_id_list) == 1:
        return [0, 0, 0], [0,0, 0, 0]

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        #print(true_masks, true_id)
        t_mask = true_masks[true_id]
        pred = pred.squeeze(0)
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id.astype(int))
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)

    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return dq * sq
    
def main(args,test_image_list):
    # change to 'combine_all' if you want to combine all targets into 1 cls
    test_dataset = Public_dataset(args,args.img_folder, args.mask_folder, test_img_list,phase='val',targets=[args.targets],if_prompt=False)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    if args.finetune_type == 'adapter' or args.finetune_type == 'vanilla':
        sam_fine_tune = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.dir_checkpoint,'checkpoint_best.pth'),num_classes=args.num_cls)
    elif args.finetune_type == 'lora':
        sam = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.sam_ckpt),num_classes=args.num_cls)
        sam_fine_tune = LoRA_Sam(args,sam,r=4).to('cuda').sam
        sam_fine_tune.load_state_dict(torch.load(args.dir_checkpoint + '/checkpoint_best.pth'), strict = False)
        
    sam_fine_tune = sam_fine_tune.to('cuda').eval()
    class_iou = torch.zeros(args.num_cls,dtype=torch.float)
    cls_dsc = torch.zeros(args.num_cls,dtype=torch.float)
    cls_pq = torch.zeros(args.num_cls,dtype=torch.float)
    cls_aji = torch.zeros(args.num_cls,dtype=torch.float)
    eps = 1e-9
    img_name_list = []
    pred_msk = []
    test_img = []
    test_gt = []
    
    dsc = 0

    for i,data in enumerate(tqdm(testloader)):
        imgs = data['image'].to('cuda')
        msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
        msks = msks.to('cuda')
        img_name_list.append(data['img_name'][0])

        with torch.no_grad():
            img_emb= sam_fine_tune.image_encoder(imgs)

            sparse_emb, dense_emb = sam_fine_tune.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
            pred_fine, _ = sam_fine_tune.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=sam_fine_tune.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb, 
                            multimask_output=True,
                          )
            
            dsc_batch = dice_coeff_multi_class(pred_fine.argmax(dim=1).cpu(), torch.squeeze(msks.long(),1).cpu().long(),args.num_cls)
            dsc+=dsc_batch
           
        pred_fine = pred_fine.argmax(dim=1)

        pred_msk.append(pred_fine.cpu())
        test_img.append(imgs.cpu())
        test_gt.append(msks.cpu())
        yhat = (pred_fine).cpu().long().flatten()
        y = msks.cpu().flatten()

        for j in range(args.num_cls):
            y_bi = y==j
            yhat_bi = yhat==j
            I = ((y_bi*yhat_bi).sum()).item()
            U = (torch.logical_or(y_bi,yhat_bi).sum()).item()
            class_iou[j] += I/(U+eps)

        for cls in range(args.num_cls):
            mask_pred_cls = ((pred_fine).cpu()==cls).float()
            mask_gt_cls = (msks.cpu()==cls).float()
            cls_dsc[cls] += dice_coeff(mask_pred_cls,mask_gt_cls).item()
            
        cls_pq[1] += get_fast_pq(pred_fine.cpu(),msks.cpu())
        cls_aji[1] += get_fast_aji(pred_fine.cpu(),msks.cpu())
        
    dsc /= (i+1)    
    cls_pq[1] /= (i+1)
    cls_aji[1] /= (i+1)
    cls_dsc /=(i+1)
    
    print('Dice Score: ', dsc)
    print('Classwise Dice Score:', cls_dsc)
    print('PQ Score: ', cls_pq)
    print('AJI Score: ', cls_aji)

    file_name = args.dir_checkpoint.split('/')[-1]
    save_folder = os.path.join('NuInsSeg_Testing',file_name)
    Path(save_folder).mkdir(parents=True,exist_ok = True)
    
    print("Saving masks...")
    for i in tqdm(range(len(pred_msk))):
        img = pred_msk[i].squeeze(0).numpy()
        img = (img*255).astype(np.uint8)
        img = Image.fromarray(img)
        mask_name = img_name_list[i].split('/')[-1]
        img.save(os.path.join(save_folder, mask_name))
        #np.save(os.path.join(save_folder,'test_name.npy'),np.concatenate(np.expand_dims(img_name_list,0),axis=0))
    
    
if __name__ == "__main__":
    args = cfg.parse_args()

    if 1: # if you want to load args from taining setting or you want to identify your own setting
        args_path = f"{args.dir_checkpoint}/args.json"

        # Reading the args from the json file
        with open(args_path, 'r') as f:
            args_dict = json.load(f)
        
        # Converting dictionary to Namespace
        args = Namespace(**args_dict)
        
    dataset_name = args.dataset_name
    print('train dataset: {}'.format(dataset_name)) 
    print(args.img_folder, args.val_img_list)
    test_img_list =  args.val_img_list
    main(args,test_img_list)