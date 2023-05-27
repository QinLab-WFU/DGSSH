
import itertools
import os
import time
import random
import argparse
import datetime
import numpy as np
import scipy.io as scio
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
# import torch.distributed as dist
from triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss, adapted_triplet_loss
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
import subset_sampler as subsetsampler
from config import get_config
from models import build_model
# from calc_hr import CalcTopMapWithPR
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, \
    TensorboardLogger, mean_average_precision_R
from lr_scheduler import build_scheduler
import torch.optim as optim
import torchvision.transforms as transforms
import data_processing as dp
from timm.data import Mixup
from torch.utils.data import DataLoader
from torch.autograd import Variable
import calc_hr as calc_hr
import adsh_loss as al
import  tqdm
import json

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
  

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def encoding_onehot(target, nclasses=30):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot


def main(config, date_str):
   
        '''
        training procedure finishes, evaluation
        '''
        model.eval()
        save_dir = '/home/admin01/桌面/论文代码/SWTH-main/log/4.6/'
        # os.makedirs(save_dir, exist_ok=True)
        # save_model = '/home/admin01/桌面/论文代码/SWTH-main/result/64-bit/WHURS/'
        # os.makedirs(save_model, exist_ok=True)
        testloader = DataLoader(dset_test, batch_size=1,
                                shuffle=False,
                                num_workers=0)
        qB = encode(model, testloader, num_test, config.MODEL.hash_length)
        rB = V
        
        #     print("pr curve save to ", config.pr_curve_path)
        mAP = calc_hr.calc_map(qB, rB, test_labels.numpy(), database_labels.numpy())
        if mAP > Best_mAP:
            Best_mAP = mAP

        # torch.save(model.state_dict(), os.path.join(save_model, "model-" + str(epoch) + ".pth"))
        logger.info('[Evaluation: mAP: %.4f]', Best_mAP)
        




