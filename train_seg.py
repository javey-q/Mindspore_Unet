import argparse
import os.path
import random

import numpy as np
import mindspore as ms
from mindspore import dataset as ds
from mindspore import nn, context

from src.Criterion import SoftmaxCrossEntropyLoss
from src.RainSegmentDataset import RainDataset, Mode
from src.se_resnext50_fpn import seresnext50_unet_fpn

seed = 1

np.random.seed(seed)
random.seed(seed)
ms.set_seed(seed)
ds.config.set_seed(seed)


def cosine_lr(base_lr, decay_steps, total_steps, resume_steps=0):
    for i in range(resume_steps, total_steps):
        step_ = min(i, decay_steps)
        yield base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))


def get_args():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--root', default='./datas', type=str)
    parser.add_argument('--epochs', default=200, type=int, help='Number of total epochs to train.')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_cls', default=21, type=int)
    parser.add_argument('--ignore_label', default=-1, type=int)
    parser.add_argument('--base_size', default=[512, 512], nargs='+', type=int)
    parser.add_argument('--crop_size', default=[256, 256], nargs='+', type=int)
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--batch_size', default=4, type=int, help='Number of datas in one batch.')
    parser.add_argument('--device_target', default='GPU', type=str, help='Ascend, GPU, CPU')
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--resume', default=None, type=str)

    return parser.parse_args()


def trainNet(args, train_model, eval_model, train_loader, valid_loader, start_epoch, max_epoch):
    for epoch in range(start_epoch, max_epoch + 1):
        train_model.set_train(True)
        train_avg_loss = 0
        for step, (imgs, masks) in enumerate(train_loader):
            train_loss = train_model(imgs, masks)
            train_avg_loss += train_loss.asnumpy() / len(train_loader)
            if step % args.print_freq == 0:
                print(f'Epoch {epoch} - {step} steps - train loss: {train_loss.asnumpy()}')

        eval_model.set_train(False)
        if epoch % args.eval_freq == 0:
            valid_avg_loss = 0
            for step, (imgs, masks, imgs_name) in enumerate(valid_loader):
                valid_loss, preds, masks = eval_model(imgs, masks)
                # TODO calculate iou

                valid_avg_loss += valid_loss.asnumpy() / len(valid_loader)

                # TODO visualize(preds, masks, imgs_name)

            print(f'''
Epoch {epoch}
    train loss: {train_avg_loss}
    valid loss: {valid_avg_loss}
            ''')

def main():
    args = get_args()
    print(args)

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    net = seresnext50_unet_fpn(args.num_cls, args.crop_size)

    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume) and args.resume.endswith('ckpt'):
            param_dict = ms.load_checkpoint(args.resume)
            ms.load_param_into_net(net, param_dict['net'])
            start_epoch = param_dict['epoch'] + 1
        else:
            raise ValueError(f'Failed to resume with \'{args.resume}\'')

    train_ds = ds.GeneratorDataset(
        source=RainDataset(
            root=args.root,
            mode=Mode.train,
            multiscale=args.multiscale
        ),
        column_names=['img', 'mask'],
        shuffle=True
    ).batch(args.batch_size)
    train_steps = train_ds.get_dataset_size()
    train_loader = train_ds.create_tuple_iterator()
    total_train_steps = train_steps * args.epochs
    lr_iter = cosine_lr(args.lr, total_train_steps, total_train_steps, train_steps * (start_epoch - 1))

    valid_ds = ds.GeneratorDataset(
        source=RainDataset(
            root=args.root,
            mode=Mode.valid,
            multiscale=args.multiscale
        ),
        column_names=['img', 'mask', 'img_name'],
        shuffle=False
    ).batch(args.batch_size)
    valid_loader = valid_ds.create_tuple_iterator()

    opt = nn.Adam(params=net.trainable_params(), learning_rate=lr_iter,
                  weight_decay=0.0005, loss_scale=1024.0)
    loss_fn = SoftmaxCrossEntropyLoss(args.num_cls, args.ignore_label)
    loss_scale_manager = ms.train.loss_scale_manager.FixedLossScaleManager(1024.0, False)
    train_model = ms.build_train_network(
        network=net, optimizer=opt, loss_fn=loss_fn,
        level='O3', boost_level='O1', loss_scale_manager=loss_scale_manager
    )
    eval_model = nn.WithEvalCell(network=net, loss_fn=loss_fn, add_cast_fp32=True)

    try:
        trainNet(args, train_model, eval_model, train_loader, valid_loader, start_epoch, args.epochs)
    except KeyboardInterrupt:
        print('Interrupted')


if __name__ == '__main__':
    main()
