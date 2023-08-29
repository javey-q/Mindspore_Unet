import mindspore as ms
import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import nn, ops
from mindspore.common import dtype
from mindspore.common.initializer import initializer, HeNormal, Normal, XavierUniform

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
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


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
            nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.SequentialCell(
            nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, padding=0, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU())
        self.drop_aspp = ops.Dropout2D(0.5)

    def construct(self, x):
        target_size = x.shape[2:]
        x0 = ops.AdaptiveAvgPool2D((1, 1))(x)
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
            init_weight(conv3x3(in_ch, out_ch * 2)),
            nn.ReLU(),
            init_weight(conv3x3(out_ch * 2, out_ch))
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


class CenterBlock(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(CenterBlock, self).__init__()
        self.conv = init_weight(conv3x3(in_channel, out_channel))

    def construct(self, inputs):
        out = self.conv(inputs)
        return out


class ChannelAttentionModule(nn.Cell):
    def __init__(self, in_channel, reduction):
        super(ChannelAttentionModule, self).__init__()
        self.fc = nn.SequentialCell(
            init_weight(conv1x1(in_channel, in_channel // reduction)),
            nn.ReLU(),
            init_weight(conv1x1(in_channel // reduction, in_channel))
        )
        self.sigmoid = nn.Sigmoid()

    def construct(self, inputs):
        x1 = self.fc(ops.ReduceMean(keep_dims=True)(inputs, (2, 3)))
        x2 = self.fc(ops.ReduceMax(keep_dims=True)(inputs, (2, 3)))
        out = self.sigmoid(x1 + x2)
        return out


class SpatialAttentionModule(nn.Cell):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv3x3 = init_weight(conv3x3(2, 1))
        self.sigmoid = nn.Sigmoid()

    def construct(self, inputs):
        avgout = ops.ReduceMean(keep_dims=True)(inputs, 1)
        maxout = ops.ReduceMax(keep_dims=True)(inputs, 1)
        out = ops.Concat(axis=1)((avgout, maxout))
        out = self.sigmoid(self.conv3x3(out))
        return out


class CBAM(nn.Cell):
    def __init__(self, in_channel, reduction):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()

    def construct(self, inputs):
        out = self.channel_attention(inputs) * inputs
        out = self.spatial_attention(out) * out
        return out


class ResizeUpsampleBlock(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(ResizeUpsampleBlock, self).__init__()
        self.conv = init_weight(conv3x3(in_channel, out_channel))

    def construct(self, inputs):
        bs, c, h, w = F.shape(inputs)
        inputs = P.ResizeBilinear((h * 2, w * 2))(inputs)
        out = self.conv(inputs)
        return out


class DecodeBlock(nn.Cell):
    def __init__(self, in_channel, out_channel, target_size, cbam: bool):
        super(DecodeBlock, self).__init__()

        self.bn1 = init_weight(nn.BatchNorm2d(in_channel))
        self.relu = nn.ReLU()
        # self.upsample = nn.Conv2dTranspose(
        #     in_channel, in_channel,
        #     kernel_size=2, stride=2, pad_mode='same'
        # )
        self.upsample = ResizeUpsampleBlock(in_channel, in_channel)
        self.conv3x3_1 = init_weight(conv3x3(in_channel, in_channel))
        self.bn2 = init_weight(nn.BatchNorm2d(in_channel))
        self.conv3x3_2 = init_weight(conv3x3(in_channel, out_channel))
        self.cbam = None
        if cbam:
            self.cbam = CBAM(out_channel, reduction=16)
        self.conv1x1 = init_weight(conv1x1(in_channel, out_channel))

    def construct(self, inputs):
        out = self.relu(self.bn1(inputs))
        out = self.upsample(out)
        out = self.conv3x3_2(self.relu(self.bn2(self.conv3x3_1(out))))
        if self.cbam is not None:
            out = self.cbam(out)
        out += self.conv1x1(self.upsample(inputs))  # shortcut
        return out


class UNET_SERESNEXT50(nn.Cell):
    def __init__(self, resolution, load_pretrained=True):
        super(UNET_SERESNEXT50, self).__init__()

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

        # center
        # self.center = CenterBlock(2048, 512)
        stride = 1
        self.center = ASPP(2048, 256, out_c=512, dilations=[stride * 1, stride * 2, stride * 3, stride * 4])

        # decoder
        self.decoder4 = DecodeBlock(512 + 2048, 64, target_size=(h // 16, w // 16), cbam=False)
        self.decoder3 = DecodeBlock(64 + 1024, 64, target_size=(h // 8, w // 8), cbam=False)
        self.decoder2 = DecodeBlock(64 + 512, 64, target_size=(h // 4, w // 4), cbam=False)
        self.decoder1 = DecodeBlock(64 + 256, 64, target_size=(h // 2, w // 2), cbam=False)
        self.decoder0 = DecodeBlock(64, 64, target_size=(h, w), cbam=False)

        # upsample
        self.upsample4 = nn.Conv2dTranspose(64, 64, kernel_size=16, stride=16, pad_mode='same')
        self.upsample3 = nn.Conv2dTranspose(64, 64, kernel_size=8, stride=8, pad_mode='same')
        self.upsample2 = nn.Conv2dTranspose(64, 64, kernel_size=4, stride=4, pad_mode='same')
        self.upsample1 = nn.Conv2dTranspose(64, 64, kernel_size=2, stride=2, pad_mode='same')
        self.concat = ms.ops.Concat(axis=1)

        self.fpn = FPN([512, 64, 64, 64, 64], [64, 64, 64, 64, 64])

        # final conv
        self.final_conv = nn.SequentialCell(
            init_weight(conv3x3(320, 64)),
            nn.ELU(),
            init_weight(conv3x3(64, 1))
        )

    def construct(self, inputs):
        # encoder
        x0 = self.encoder0(inputs)  # ->(*,64,h/2,w/2)
        x1 = self.encoder1(x0)  # ->(*,256,h/4,w/4)
        x2 = self.encoder2(x1)  # ->(*,512,h/8,w/8)
        x3 = self.encoder3(x2)  # ->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3)  # ->(*,2048,h/32,w/32)

        # center
        y5 = self.center(x4)  # (*, 512, h/32, w/32)

        # decoder
        y4 = self.decoder4(self.concat((x4, y5)))
        y3 = self.decoder3(self.concat((x3, y4)))
        y2 = self.decoder2(self.concat((x2, y3)))
        y1 = self.decoder1(self.concat((x1, y2)))

        # hypercolumns
        # y4 = self.upsample4(y4)
        # y3 = self.upsample3(y3)
        # y2 = self.upsample2(y2)
        # y1 = self.upsample1(y1)
        # hypercol = self.concat((y0, y1, y2, y3, y4))
        hypercol = self.fpn([y5, y4, y3, y2], y1)

        logits = self.final_conv(hypercol)
        bs, c, h, w = F.shape(logits)
        logits = P.ResizeBilinear((h * 2, w * 2))(logits)

        return logits


def seresnext50_unet(resolution=(512, 512), load_pretrained=False):
    model = UNET_SERESNEXT50(resolution, load_pretrained)
    return model
