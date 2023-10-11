# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import itertools
import os
import time
import random
import argparse
import datetime
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import util.subset_sampler as subsetsampler
from config import get_config
from models import build_model
from util.logger import create_logger
from util.utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper
from util.learning_scheme import build_scheduler
import torch.optim as optim
import torchvision.transforms as transforms
import util.process_datas as pd
from torch.utils.data import DataLoader
from loss.asy_loss import *
from loss.triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss, corrective_triplet_loss
from evaluate import calc_map



try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Deep Global Semantic Structure-preserving Hashing', add_help=False)
    parser.add_argument('--cfg', type=str, default='./configs/swin_config.yaml', metavar="FILE",
                        help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default='./dataset/AID/', help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained', default='./pretrained/swin_pre.pth',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, default=2, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='./result/', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--triplet_strategy', default='batch_corrective', type=str,
                        help='triplet_strategy')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def encoding_onehot(target, nclasses=30):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot


def _dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dset_database = pd.ProcessingUCMD_21(
        'dataset/UCMD', 'database_index_img.txt', 'database_index_label.txt', transformations)
    dset_test = pd.ProcessingUCMD_21(
        'dataset/UCMD', 'test_index_img.txt', 'test_index_label.txt', transformations)
    num_database, num_test = len(dset_database), len(dset_test)

    def load_label(filename, DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        fp = open(path, 'r')
        labels = [x.strip() for x in fp]
        fp.close()
        return torch.LongTensor(list(map(int, labels)))

    testlabels = load_label('test_index_label.txt', 'dataset/UCMD')
    databaselabels = load_label('database_index_label.txt', 'dataset/UCMD')

    testlabels = encoding_onehot(testlabels)
    databaselabels = encoding_onehot(databaselabels)

    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    labels = (databaselabels, testlabels)

    return nums, dsets, labels


def calc_sim(database_label, train_label):
    S = (database_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    '''
    soft constraint
    '''
    r = S.sum() / (1 - S).sum()
    S = S * (1 + r) - r
    return S


def calc_loss(u, V, U,select_index, beta, target,num_class):
    num_database = V.shape[0]
    similarity_loss = similarityloss(u, target,num_class)
    V_omega = V[select_index, :]
    constraint_loss = (U - V_omega) ** 2
    loss = (similarity_loss.sum() + beta * constraint_loss.sum()) / (num_database * num_database)
    return loss


def encode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda())
        output, output_s, output_t = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
    return B

def main(config):
    # TODO data_loader_database; tensorboard
    '''
     dataset preprocessing
     '''
    nums, dsets, labels = _dataset()
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    model.train()
    start = time.time()
    logger.info(str(model))

    optimizer = optim.AdamW(model.parameters(), eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                            lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    if config.AMP_OPT_LEVEL != "O0":  # 不执行
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model_without_ddp = model
    lr_scheduler = build_scheduler(config, optimizer, num_database)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    asymmetric_loss = AsyLoss(config.MODEL.beta_param, config.MODEL.hash_length, num_database)
    if config.AUG.MIXUP > 0.:
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
    V = np.zeros((num_database, config.MODEL.hash_length))
    logger.info("Start training")
    Best_mAP = 0.0
    start_time = time.time()
    mAP = 0.0
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        select_index = list(np.random.permutation(range(num_database)))[0: num_database]
        _sampler = subsetsampler.SubsetSampler(select_index)
        trainloader = DataLoader(dset_database, batch_size=32,
                                 sampler=_sampler,
                                 shuffle=False,
                                 num_workers=0)
        sample_label = database_labels.index_select(0, torch.from_numpy(np.array(select_index)))
        Sim = calc_sim(sample_label, database_labels)
        U = np.zeros((num_database, config.MODEL.hash_length), dtype=np.float64)
        loss_cortriplet = 0
        for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
            batch_size_ = train_label.size(0)
            u_ind = np.linspace(iteration * 32, np.min((num_database, (iteration + 1) * 32)) - 1,
                                batch_size_, dtype=int)

            train_input = Variable(train_input.cuda())
            train_label = train_label.cuda()
            train_label = train_label.squeeze(1)

            output, output_s, output_t = model(train_input)
            # triplet loss
            embeddings = torch.nn.functional.normalize(output_t, p=2, dim=1, eps=1e-10)
            if config.TRAIN.TRIPLET_STRATEGY == "batch_all":
                loss_cortriplet = batch_all_triplet_loss(train_label, embeddings, config.MODEL.margin, squared=False)
            elif config.TRAIN.TRIPLET_STRATEGY == "batch_hard":
                loss_cortriplet = batch_hard_triplet_loss(train_label, embeddings, config.MODEL.margin, squared=False)
            elif config.TRAIN.TRIPLET_STRATEGY == "batch_adaptive":
                loss_cortriplet = corrective_triplet_loss(train_label, embeddings, config.MODEL.margin, squared=False)
            loss_cortriplet += loss_cortriplet
            loss_cla = criterion(output_s, train_label)

            S = Sim.index_select(0, torch.from_numpy(u_ind))
            U[u_ind, :] = output.cpu().data.numpy()
            model.zero_grad()
            loss_asy = asymmetric_loss(output, V[batch_ind.cpu().numpy(), :], train_label,config.MODEL.NUM_CLASSES)
            loss = loss_asy + config.MODEL.alph_param * loss_cla + loss_cortriplet
            loss.backward()
            optimizer.step()
        lr_scheduler.step_update(epoch * num_database + iteration)
        '''
                learning binary codes: discrete coding
                '''
        barU = np.zeros((num_database, config.MODEL.hash_length))
        barU[select_index, :] = U
        Q = -2 * config.MODEL.hash_length * Sim.cpu().numpy().transpose().dot(U) - 2 * config.MODEL.beta_param * barU

        for k in range(config.MODEL.hash_length):
            sel_ind = np.setdiff1d([ii for ii in range(config.MODEL.hash_length)], k)
            V_ = V[:, sel_ind]
            Uk = U[:, k]
            U_ = U[:, sel_ind]
            V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk)))
        loss_ = calc_loss(output, V, U,select_index,config.MODEL.beta_param, train_label,config.MODEL.NUM_CLASSES)

        logger.info('[epoch: %3d][Train Loss: %.4f]', epoch, loss_)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))
        '''
        training procedure finishes, evaluation
        '''
        model.eval()
        testloader = DataLoader(dset_test, batch_size=1,
                                shuffle=False,
                                num_workers=0)

        qB = encode(model, testloader, num_test, config.MODEL.hash_length)
        rB = V
        mAP = calc_map(qB, rB, test_labels.numpy(), database_labels.numpy())
        if mAP > Best_mAP:
            Best_mAP = mAP
        logger.info('[Evaluation: mAP: %.4f]', Best_mAP)
if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"
    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()

    config.MODEL.alph_param = config.MODEL.alph_param
    config.MODEL.beta_param = config.MODEL.beta_param
    config.MODEL.hash_length = config.MODEL.hash_length

    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr

    date_str = '/' + str(config.MODEL.hash_length) + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    config.OUTPUT = config.OUTPUT + config.DATA.DATASET + date_str

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")
    # if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    # print config
    logger.info(config.dump())


    main(config)



