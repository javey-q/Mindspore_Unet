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

from src.Criterion import BCE_DICE_LOSS, CrossEntropyWithLogits, BCE_DICE_LOSSv2
from src.RemoteSensingDataset import RSDataset, Mode
from src.se_resnext50 import seresnext50_unet
from src.se_resnext50_fpn import seresnext50_unet_fpn

seed = 1

np.random.seed(seed)
random.seed(seed)
ms.set_seed(seed)
ds.config.set_seed(seed)

visual_flag = True

# net_name = 'seresnext50_unet'
net_name = 'seresnext50_unet_fpn'

crop_size = 1280
dir_root = '../dataset_1920'
dir_weight = './weights/seresnext50_unet_fpn_best.ckpt'
dir_log = './logs'
dir_save_pred = f'../dataset_1920/valid/pred'
prefix = net_name
python_multiprocessing = True
num_parallel_workers = 16
FixedLossScaleManager = 1024.0


def calc_iou(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score * 100



def evalNet(net, batch_size):
    if (not os.path.isfile(dir_weight)) and dir_weight.endswith('.ckpt'):
        raise ValueError('check out the path of weight file')

    param_dict = ms.load_checkpoint(dir_weight)
    ms.load_param_into_net(_net, param_dict)
    dataset_valid_buffer = RSDataset(root=dir_root, mode=Mode.valid,
                                     multiscale=False,
                                     crop_size=(crop_size, crop_size))
    dataset_valid = ds.GeneratorDataset(
        source=dataset_valid_buffer,
        column_names=['data', 'label','filename'],
        shuffle=False,
        python_multiprocessing=python_multiprocessing,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=16
    )
    dataset_valid = dataset_valid.batch(batch_size)
    valid_steps = dataset_valid.get_dataset_size()
    dataloader_valid = dataset_valid.create_tuple_iterator(num_epochs=1, output_numpy=True)

    logger.info(f'''
==================================DATA=======================================
    Dataset:
        batch_size: {batch_size}
        crop_size : {crop_size}
        valid:
            nums : {len(dataset_valid_buffer)}
            steps: {valid_steps}
=============================================================================
        ''')
    criterion_valid = BCE_DICE_LOSS()
    eval_model = nn.WithEvalCell(network=net, loss_fn=criterion_valid, add_cast_fp32=True)

    logger.info(f'Begin eval:')
        # eval
    eval_model.set_train(False)
    valid_avg_loss = 0
    valid_avg_iou = 0
    with tqdm(total=valid_steps, desc='Validation', unit='batch') as eval_pbar:
        for idx, (imgs, masks, filenames) in enumerate(dataloader_valid):
            imgs = ms.Tensor(imgs, dtype=ms.float32)
            masks = ms.Tensor(masks, dtype=ms.float32)
            valid_loss, preds, masks = eval_model(imgs, masks)

            pred_buffer = preds.asnumpy().copy()
            pred_buffer[pred_buffer >= 0] = 1
            pred_buffer[pred_buffer < 0] = 0
            mask_buffer = masks.asnumpy().copy()

            if visual_flag:
                for i in range(pred_buffer.shape[0]):
                    filename = filenames[i].astype(str)
                    visual_pred = pred_buffer[i, 0, :, :].astype(np.uint8)
                    # visual_mask = mask_buffer[i, 0, :, :].astype(np.uint8)
                    if not os.path.exists(dir_save_pred):
                        os.mkdir(dir_save_pred)
                    cv2.imwrite(f'{dir_save_pred}/{filename}', visual_pred * 255)
                    # cv2.imwrite(f'{dir_buffer}/{idx}_{i}_mask.png', visual_mask * 255)

            iou_score = calc_iou(mask_buffer, pred_buffer)
            valid_avg_iou += iou_score / valid_steps
            valid_avg_loss += valid_loss / valid_steps

            eval_pbar.update(1)
            eval_pbar.set_postfix(**{'IoU (batch)': iou_score})

    logger.info(f'''
            validation loss : {valid_avg_loss}
            validation iou  : {valid_avg_iou}
            ''')


    logger.info('eval finished.')


def get_args():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--root', default='../dataset_1920', type=str)
    parser.add_argument('--batch_size', default=8, type=int, help='Number of datas in one batch.')
    parser.add_argument('--device_target', default='Ascend', type=str)
    parser.add_argument('--num_parallel_workers', default=32, type=int)
    parser.add_argument('--close_python_multiprocessing', default=False, action='store_true')
    parser.add_argument('--visual', default=False, action='store_true', help='Visual at eval.')
    parser.add_argument('--load_weight', default=None, type=str)
    parser.add_argument('--loss', default=None, type=str)

    return parser.parse_args()


def init_logger():
    fmt = '%(asctime)s - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(level=logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(filename=f'{dir_log}/eval.log', mode='w')
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

    if net_name == 'seresnext50_unet':
        _net = seresnext50_unet_fpn(
            resolution=(crop_size, crop_size),
        )
    elif net_name == 'seresnext50_unet_fpn':
        _net = seresnext50_unet_fpn(
            resolution=(crop_size, crop_size),
        )

    if args.load_weight is not None:
        dir_weight = args.load_weight

        
    if args.close_python_multiprocessing:
        python_multiprocessing = False

    if args.visual:
        visual_flag = True
        

    logger.info(f'''
==================================INFO=======================================
    path config :
        data_root   : {dir_root}
        dir_weights : {dir_weight}
        dir_log     : {dir_log}
    
    net : {net_name}
    training config :
        batch_size      : {args.batch_size}
        device          : {args.device_target}
        multiprocessing : {'Enabled' if python_multiprocessing else 'Disabled'}
        visual in eval  : {'Enabled' if visual_flag else 'Disabled'}
=============================================================================
    ''')

    try:
        evalNet(
            net=_net,
            batch_size=args.batch_size
        )
    except InterruptedError:
        logger.error('Interrupted')
