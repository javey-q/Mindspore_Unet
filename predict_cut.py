import logging

import argparse

import os.path
import time

import cv2
import mindspore as ms
import mindspore.dataset as ds
import numpy as np
import mindspore.ops.functional as F
from mindspore.dataset import context
from mindspore.nn import Adam, WithEvalCell
from tqdm import tqdm

from src.Criterion import BCE_DICE_LOSS
from src.RemoteSensingDataset import RSDataset, Mode
from src.se_resnext50 import seresnext50_unet
from src.se_resnext50_fpn import seresnext50_unet_fpn

net_name = 'seresnext50_unet'

batch_size = 8
dir_root = '../dataset_1920/test'
dir_weight = './weights/seresnext50_unet_best.ckpt'
dir_pred = '../dataset_1920/test/pred'
dir_log = './logs'
figsize = 1280
python_multiprocessing = True
num_parallel_workers = 16

if not os.path.exists(dir_pred):
    os.makedirs(dir_pred)

def predictNet(net):
    # model = WithEvalCell(network=net, loss_fn=BCE_DICE_LOSS(), add_cast_fp32=True)
    # model.set_train(False)
    model = net.to_float(ms.dtype.float16)
    dataset_predict_buffer = RSDataset(root=dir_root, mode=Mode.predict_cut,
                                       multiscale=False,
                                       crop_size=(figsize, figsize))
    dataset_predict = ds.GeneratorDataset(
        source=dataset_predict_buffer,
        column_names=['data', 'original_shape', 'filename'],
        shuffle=False, num_parallel_workers=num_parallel_workers,
        python_multiprocessing=python_multiprocessing,
        max_rowsize=32
    )
    dataset_predict = dataset_predict.batch(batch_size)
    predict_steps = dataset_predict.get_dataset_size()
    dataloader_predict = dataset_predict.create_tuple_iterator(num_epochs=1, output_numpy=True)
    with tqdm(total=predict_steps, desc='Prediction', unit='batch') as pbar:
        for step, (imgs, original_shapes, filenames) in enumerate(dataloader_predict):
            imgs = ms.Tensor(imgs, dtype=ms.float16)
            bs, _, _, _ = F.shape(imgs)
            preds = model(imgs)
            preds = preds.asnumpy()
            for i in range(bs):
                original_shape = original_shapes[i]
                filename = filenames[i].astype(str)
                pred = preds[i,0,:,:]
                pred = cv2.resize(pred.astype(np.float32), (original_shape[1], original_shape[0]))

                pred[pred >= 0] = 255
                pred[pred < 0] = 0
                pred = pred.astype(np.uint8)
                cv2.imwrite(f'{dir_pred}/{filename}', pred)

            pbar.update(1)


def get_args():
    parser = argparse.ArgumentParser(description='Prediction')

    parser.add_argument('--root', default=None, type=str)
    parser.add_argument('--device_target', default='Ascend', type=str)
    parser.add_argument('--figsize', default=None, type=int)
    parser.add_argument('--dir_pred', default=None, type=str)
    parser.add_argument('--load_weight', default=None, type=str)
    parser.add_argument('--num_parallel_workers', default=None, type=int)
    parser.add_argument('--close_python_multiprocessing', default=False, action='store_true')

    return parser.parse_args()


def init_logger():
    fmt = '%(asctime)s - %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(level=logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(filename=f'{dir_log}/predict.log', mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


if __name__ == '__main__':
    logger = logging.getLogger()
    init_logger()

    args = get_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    if args.root is not None:
        dir_root = args.root

    if args.dir_pred is not None:
        dir_pred = args.dir_pred

    if args.num_parallel_workers is not None:
        num_parallel_workers = args.num_parallel_workers

    if args.close_python_multiprocessing:
        python_multiprocessing = False

    if args.figsize is not None:
        figsize = args.figsize

    # _net = seresnext50_unet(
    #     resolution=(figsize, figsize),
    #     load_pretrained=False
    # )
    _net = seresnext50_unet_fpn(
        resolution=(figsize, figsize),
        load_pretrained=False
    )

    if args.load_weight is not None:
        dir_weight = args.load_weight

    if (not os.path.isfile(dir_weight)) and dir_weight.endswith('.ckpt'):
        raise ValueError('check out the path of weight file')

    param_dict = ms.load_checkpoint(dir_weight)
    ms.load_param_into_net(_net, param_dict)

    logger.info(f'''
=============================================================================
    path config :
        data_root   : {dir_root}   
        dir_pred    : {dir_pred}
        dir_log     : {dir_log}  

    net : {net_name}
        weight              : {dir_weight}

    predict config :
        figsize         : {figsize}
        device          : {args.device_target}
        multiprocessing : {'Enabled' if python_multiprocessing else 'Disabled'}
=============================================================================
    ''')

    try:
        predictNet(net=_net)
    except InterruptedError:
        logger.error('Interrupted')
