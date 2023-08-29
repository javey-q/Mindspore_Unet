import argparse
import ast
import logging
import os.path
import random

import cv2
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, context
from tqdm import tqdm
import mindspore.ops as F
import mindspore.ops.operations as P

from src.Criterion import BCE_DICE_LOSS, CrossEntropyWithLogits,BCE_DICE_LOSS
from src.RemoteSensingDataset import RSDataset, Mode
from src.se_resnext50 import seresnext50_unet
from src.se_resnext50_fpn import seresnext50_unet_fpn

seed = 1

np.random.seed(seed)
random.seed(seed)
ms.set_seed(seed)
ds.config.set_seed(seed)

visual_flag = False

# net_name = 'seresnext50_unet'
net_name = 'seresnext50_unet_fpn'


resume_epoch = 1
base_size = 800
crop_size = 640
dir_root = './datas'
dir_weights = './weights'
dir_log = './logs'
prefix = net_name
python_multiprocessing = True
num_parallel_workers = 16
eval_per_epoch = 0
FixedLossScaleManager = 1024.0


def calc_iou(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score * 100


def cosine_lr(base_lr, decay_steps, total_steps, resume_steps=0):
    for i in range(resume_steps, total_steps):
        step_ = min(i, decay_steps)
        yield base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))


def trainNet(net, criterion, epochs, batch_size):
    dataset_train_buffer = RSDataset(root=dir_root, mode=Mode.train,
                                     multiscale=True, scale=0.5,
                                     base_size=base_size, crop_size=(crop_size, crop_size))
    dataset_train = ds.GeneratorDataset(
        source=dataset_train_buffer,
        column_names=['data', 'label'],
        shuffle=True,
        python_multiprocessing=python_multiprocessing,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=16
    )
    dataset_train = dataset_train.batch(batch_size)
    train_steps = dataset_train.get_dataset_size()
    dataloader_train = dataset_train.create_tuple_iterator()

    dataset_valid_buffer = RSDataset(root=dir_root, mode=Mode.valid,
                                     multiscale=False,
                                     crop_size=(crop_size, crop_size))
    dataset_valid = ds.GeneratorDataset(
        source=dataset_valid_buffer,
        column_names=['data', 'label'],
        shuffle=False,
        python_multiprocessing=python_multiprocessing,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=16
    )
    dataset_valid = dataset_valid.batch(batch_size)
    valid_steps = dataset_valid.get_dataset_size()
    dataloader_valid = dataset_valid.create_tuple_iterator()

    logger.info(f'''
==================================DATA=======================================
    Dataset:
        batch_size: {batch_size}
        base_size : {base_size}
        crop_size : {crop_size}
        train:
            nums : {len(dataset_train_buffer)}
            steps: {train_steps}
        valid:
            nums : {len(dataset_valid_buffer)}
            steps: {valid_steps}
=============================================================================
        ''')

    # net_with_loss = nn.WithLossCell(backbone=net, loss_fn=criterion)
    #
    # train_model = TrainOneStepCell(network=net_with_loss, optimizer=opt)

    total_train_steps = train_steps * epochs
    resume_steps = train_steps * (resume_epoch - 1)
    lr_iter = cosine_lr(0.0002, total_train_steps, total_train_steps, resume_steps)

    params = net.trainable_params()
    opt = nn.Adam(params=params, learning_rate=lr_iter, weight_decay=0.0005, loss_scale=FixedLossScaleManager)

    loss_scale_manager = ms.train.loss_scale_manager.FixedLossScaleManager(FixedLossScaleManager, False)
    train_model = ms.build_train_network(network=net, optimizer=opt, loss_fn=criterion,
                                         level='O3', boost_level='O1', loss_scale_manager=loss_scale_manager)

    eval_model = nn.WithEvalCell(network=net, loss_fn=criterion, add_cast_fp32=True)

    logger.info(f'Begin training:')

    best_model_epoch = 0
    best_valid_iou = 0
    for epoch in range(resume_epoch, epochs + 1):
        # train
        train_model.set_train(True)
        train_avg_loss = 0
        with tqdm(total=train_steps, desc=f'Epoch {epoch}/{epochs}', unit='batch') as train_pbar:
            for step, (imgs, masks) in enumerate(dataloader_train):
                train_loss = train_model(imgs, masks)
                train_avg_loss += train_loss.asnumpy() / train_steps

                train_pbar.update(1)
                train_pbar.set_postfix(**{'loss (batch)': train_loss.asnumpy()})

        # eval
        eval_model.set_train(False)
        if eval_per_epoch == 0 or epoch % eval_per_epoch == 0:
            valid_avg_loss = 0
            valid_avg_iou = 0
            with tqdm(total=valid_steps, desc='Validation', unit='batch') as eval_pbar:
                for idx, (imgs, masks) in enumerate(dataloader_valid):
                    valid_loss, preds, masks = eval_model(imgs, masks)
                    bs, c, h, w = F.shape(preds)
                    pred_buffer = preds.asnumpy().copy()
                    pred_buffer[pred_buffer >= 0] = 1
                    pred_buffer[pred_buffer < 0] = 0
                    mask_buffer = masks.asnumpy().copy()

                    if visual_flag:
                        for i in range(pred_buffer.shape[0]):
                            visual_pred = pred_buffer[i, 0, :, :].astype(np.uint8)
                            visual_mask = mask_buffer[i, 0, :, :].astype(np.uint8)
                            dir_buffer = f'./valid_buffer/{epoch}'
                            if not os.path.exists(dir_buffer):
                                os.mkdir(dir_buffer)
                            cv2.imwrite(f'{dir_buffer}/{idx}_{i}_pred.png', visual_pred * 255)
                            cv2.imwrite(f'{dir_buffer}/{idx}_{i}_mask.png', visual_mask * 255)

                    iou_score = calc_iou(mask_buffer, pred_buffer)
                    valid_avg_iou += iou_score / valid_steps
                    valid_avg_loss += valid_loss / valid_steps

                    eval_pbar.update(1)
                    eval_pbar.set_postfix(**{'IoU (batch)': iou_score})

            if best_valid_iou is None or best_valid_iou < valid_avg_iou:
                best_valid_iou = valid_avg_iou
                best_model_epoch = epoch
                ms.save_checkpoint(net, f'{dir_weights}/{prefix}_best.ckpt')

            logger.info(f'''
    In {epoch} epoch:
            train loss      : {train_avg_loss}
            validation loss : {valid_avg_loss}
            validation iou  : {valid_avg_iou}
            best valid iou  : {best_valid_iou}
            best model saved at {best_model_epoch} epoch.
            ''')

        ms.save_checkpoint(net, f'{dir_weights}/{prefix}_last.ckpt')

    logger.info('Training finished.')


def get_args():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--root', default='./datas', type=str)
    parser.add_argument('--epochs', default=200, type=int, help='Number of total epochs to train.')
    parser.add_argument('--batch_size', default=8, type=int, help='Number of datas in one batch.')
    parser.add_argument('--device_target', default='Ascend', type=str)
    parser.add_argument('--load_pretrained', default=True, type=ast.literal_eval)
    parser.add_argument('--num_parallel_workers', default=32, type=int)
    parser.add_argument('--eval_per_epoch', default=0, type=int)
    parser.add_argument('--close_python_multiprocessing', default=False, action='store_true')
    parser.add_argument('--visual', default=False, action='store_true', help='Visual at eval.')
    parser.add_argument('--resume_epoch', default=None, type=int)
    parser.add_argument('--resume_weight', default=None, type=str)
    parser.add_argument('--loss', default=None, type=str)

    return parser.parse_args()


def init_logger():
    fmt = '%(asctime)s - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(level=logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(filename=f'{dir_log}/train.log', mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


if __name__ == '__main__':
    logger = logging.getLogger()
    init_logger()

    args = get_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)  # GRAPH_MODE

    if args.root:
        dir_root = args.root

    if args.num_parallel_workers:
        num_parallel_workers = args.num_parallel_workers

    if args.eval_per_epoch:
        eval_per_epoch = args.eval_per_epoch

    if net_name == 'seresnext50_unet':
        _net = seresnext50_unet_fpn(
            resolution=(crop_size, crop_size),
            load_pretrained=args.load_pretrained
        )
    elif net_name == 'seresnext50_unet_fpn':
        _net = seresnext50_unet_fpn(
            resolution=(crop_size, crop_size),
            load_pretrained=args.load_pretrained
        )

    if args.loss == 'BCE_Lovasz':
            _criterion = BCE_Lovasz_LOSS()
    else:
        _criterion = BCE_DICE_LOSS()


    if args.resume_epoch is not None:
        if args.resume_weight is None:
            raise ValueError('resume weights file is not define')
        dir_resume = args.resume_weight
        param_dict = ms.load_checkpoint(dir_resume)
        ms.load_param_into_net(_net, param_dict)
        resume_epoch = args.resume_epoch
        
    if args.close_python_multiprocessing:
        python_multiprocessing = False

    if args.visual:
        visual_flag = True
        

    logger.info(f'''
==================================INFO=======================================
    path config :
        data_root   : {dir_root}
        dir_weights : {dir_weights}
        dir_log     : {dir_log}
    
    net : {net_name}
        pretrained weight   : {'Enabled' if args.load_pretrained else 'Disabled'}
    
    training config :
        epochs          : {args.epochs}
        batch_size      : {args.batch_size}
        device          : {args.device_target}
        multiprocessing : {'Enabled' if python_multiprocessing else 'Disabled'}
        visual in eval  : {'Enabled' if visual_flag else 'Disabled'}
=============================================================================
    ''')

    try:
        trainNet(
            net=_net,
            criterion=_criterion,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    except InterruptedError:
        logger.error('Interrupted')
