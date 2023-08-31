import mindspore as ms
import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import nn, ops
from mindspore.common import dtype
from mindspore.common.initializer import initializer, HeNormal, Normal, XavierUniform
from mindspore.common import initializer
from mindspore.nn import SyncBatchNorm, BatchNorm2d
import math
from src.senet_ms import se_resnext50_32x4d


def conv3x3(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=1, pad_mode='pad', padding=1, dilation=1, has_bias=False)


def conv1x1(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=1, pad_mode='same', padding=0, dilation=1, has_bias=False)


def init_weight(m: nn.Cell):
    for _, cell in m.name_cells():
        if isinstance(cell, nn.Conv2d):
            initializer(HeNormal(mode='fan_in', nonlinearity='relu'), cell.weight, dtype.float32)
            if cell.bias is not None:
                cell.bias.data.zero_()
        elif isinstance(cell, nn.BatchNorm2d):
            initializer(Normal(1, 0.02), cell.weight, dtype.float32)
            cell.bias.data.zero_()
        else:
            initializer(XavierUniform(), cell.weight, dtype.float32)

    return m


class _ASPPModule(nn.Cell):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, group=1):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, pad_mode='pad', dilation=dilation, has_bias=False,
                                     group=group)
        self.bn = nn.BatchNorm2d(planes)
        self.gelu = nn.GELU()

    def construct(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.gelu(x)


class ASPP(nn.Cell):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super(ASPP, self).__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
                     [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, group=4) for d in dilations]
        self.nums = len(self.aspps)
        self.aspps = nn.CellList(self.aspps)
        self.concat = ops.Concat(axis=1)
        self.global_pool = nn.SequentialCell(
            # nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_c, 1, stride=1, padding=0, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.SequentialCell(
            nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, padding=0, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(out_c), nn.GELU())
        self.drop_aspp = nn.Dropout(0.5)

    def construct(self, x):
        target_size = x.shape[2:]
        x0 = ops.ReduceMean(keep_dims=True)(x, (2, 3))
        x0 = self.global_pool(x0)
        x0 = ops.ResizeBilinear(target_size)(x0)
        xs = []
        for i in range(self.nums):
            xs.append(self.aspps[i](x))
        # xs = [aspp(x) for aspp in self.aspps]
        x = self.concat([x0] + xs)
        out = self.out_conv(x)
        out = self.drop_aspp(out)
        return out


class FPN(nn.Cell):
    def __init__(self, in_channels: list, out_channels: list):
        super(FPN, self).__init__()
        self.convs = nn.CellList([nn.SequentialCell(
            conv3x3(in_ch, out_ch * 2),
            nn.BatchNorm2d(out_ch * 2),
            nn.GELU(),
            conv3x3(out_ch * 2, out_ch)
        ) for in_ch, out_ch in zip(in_channels, out_channels)])

    def construct(self, xs: list, last_layer):
        b, c, h, w = F.shape(last_layer)
        num = len(xs)
        hcs = []
        for i in range(num):
            xs[i] = P.ResizeBilinear((h, w))(xs[i])
            hcs.append(self.convs[i](xs[i]))
        hcs.append(last_layer)
        return ops.Concat(axis=1)(hcs)


class DecodeBlock(nn.Cell):
    def __init__(self, in_channel_up, in_channel_left, out_channel):
        super(DecodeBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channel_left)
        self.gelu = nn.GELU()
        self.upsample = ops.DepthToSpace(2)
        in_channel = in_channel_up // 4 + in_channel_left
        # self.upsample = nn.Conv2dTranspose(in_channel_up, in_channel_up//2, kernel_size=2, stride=2, pad_mode='same')
        # in_channel = in_channel_up // 2 + in_channel_left
        self.conv3x3_1 = conv3x3(in_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3x3_2 = conv3x3(out_channel, out_channel)
        self.concat = ops.Concat(axis=1)

    def construct(self, inputs_up, inputs_left):
        up_out = self.upsample(inputs_up)
        cat_x = self.gelu(self.concat([up_out, self.bn1(inputs_left)]))
        out = self.conv3x3_2(self.conv3x3_1(cat_x))

        return out


class UNET_SERESNEXT50_FPN(nn.Cell):
    def __init__(self, num_cls, resolution, load_pretrained=True):
        super(UNET_SERESNEXT50_FPN, self).__init__()

        self.num_cls = num_cls
        h, w = resolution

        seresnext50 = se_resnext50_32x4d()
        if load_pretrained:
            param_dict = ms.load_checkpoint(
                'pretrained/seresnext50_ascend_v130_imagenet2012_research_cv_top1acc79_top5acc94.ckpt')
            ms.load_param_into_net(seresnext50, param_dict)

        # encoder
        self.encoder0 = nn.SequentialCell(
            seresnext50.layer0[0],  # Conv2d
            seresnext50.layer0[1],  # BatchNorm2d
            seresnext50.layer0[2]  # ReLU
        )
        self.encoder1 = nn.SequentialCell(
            seresnext50.layer0[3],  # MaxPool2d
            seresnext50.layer1
        )
        self.encoder2 = seresnext50.layer2
        self.encoder3 = seresnext50.layer3
        self.encoder4 = seresnext50.layer4

        stride = 1
        self.aspp = ASPP(2048, 256, out_c=512, dilations=[stride * 1, stride * 2, stride * 3, stride * 4])

        # decoder
        self.decoder4 = DecodeBlock(512, 1024, 256)
        self.decoder3 = DecodeBlock(256, 512, 128)
        self.decoder2 = DecodeBlock(128, 256, 64)
        self.decoder1 = DecodeBlock(64, 64, 32)

        self.fpn = FPN([512, 256, 128, 64], [32, 32, 32, 32])

        self.drop_final = nn.Dropout(0.9)
        # final conv
        self.final_conv = nn.SequentialCell(
            conv3x3(32 * 4 + 32, 64),
            # nn.Conv2dTranspose(64, 32, kernel_size=4, stride=2, pad_mode='same'),
            # nn.BatchNorm2d(32),
            nn.GELU(),
            conv1x1(64, num_cls)
        )

    def construct(self, inputs):
        # encoder
        x0 = self.encoder0(inputs)  # ->(*,64,h/2,w/2)
        x1 = self.encoder1(x0)  # ->(*,256,h/4,w/4)
        x2 = self.encoder2(x1)  # ->(*,512,h/8,w/8)
        x3 = self.encoder3(x2)  # ->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3)  # ->(*,2048,h/32,w/32)

        # center
        y5 = self.aspp(x4)  # (*, 512, h/32, w/32)

        # decoder
        y4 = self.decoder4(y5, x3)  # 256,h/16,w/16
        y3 = self.decoder3(y4, x2)  # 128,h/8,w/8
        y2 = self.decoder2(y3, x1)  # 64,h/4,w/4
        y1 = self.decoder1(y2, x0)  # 32,h/2,w/2

        hypercol = self.fpn([y5, y4, y3, y2], y1)

        hypercol = self.drop_final(hypercol)
        logits = self.final_conv(hypercol)

        bs, c, h, w = F.shape(logits)
        logits = P.ResizeBilinear((h * 2, w * 2))(logits)

        return logits


def seresnext50_unet_fpn(num_cls, resolution=(512, 512), load_pretrained=False):
    model = UNET_SERESNEXT50_FPN(num_cls, resolution, load_pretrained)
    return model
