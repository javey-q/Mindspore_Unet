import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops as F
from mindspore.nn.loss.loss import LossBase
from mindspore import dtype as mstype, Tensor
from mindspore.nn import Cell
from mindspore import Parameter


class BCE_DICE_LOSS(Cell):
    def __init__(self):
        super(BCE_DICE_LOSS, self).__init__()
        self.sig = nn.Sigmoid()
        self.c1 = nn.BCELoss(reduction='mean')
        self.c2 = nn.DiceLoss()

    def construct(self, logits, labels):
        logits = self.sig(logits)
        loss1 = self.c1(logits, labels)
        loss2 = self.c2(logits, labels)
        return loss1 + loss2

class BCE_DICE_LOSSv2(Cell):
    def __init__(self):
        super(BCE_DICE_LOSSv2, self).__init__()
        self.sig = nn.Sigmoid()
        self.c1 = nn.BCELoss(reduction='none')
        self.c2 = nn.DiceLoss()

    def construct(self, logits, labels, edge):
        edge_weight = 4.
        logits = self.sig(logits)
        loss_bce = self.c1(logits, labels)
        edge[edge == 0] = 1.
        edge[edge == 255] = edge_weight
        loss_bce *= edge
        loss_bce = loss_bce.mean()
        loss2 = self.c2(logits, labels)
        return loss_bce + loss2
    
class BCE_Lovasz_LOSS(Cell):
    def __init__(self):
        super(BCE_Lovasz_LOSS, self).__init__()
        self.sig = nn.Sigmoid()
        self.bce = nn.BCELoss(reduction='mean')
        self.lovasz = Lovasz_hinge()

    def construct(self, logits, labels):
        lovasz_loss = self.lovasz(logits, labels)
        logits = self.sig(logits)
        loss_bce = self.bce(logits, labels)
        return lovasz_loss + loss_bce


class CrossEntropyWithLogits(nn.LossBase):
    """
    Cross-entropy loss function for semantic segmentation,
    and different classes have the same weight.
    """

    def __init__(self, num_classes=19, ignore_label=255, image_size=None):
        super(CrossEntropyWithLogits, self).__init__()
        # self.resize = F.ResizeBilinear(image_size)
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.argmax = P.Argmax(output_type=mstype.int32)
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        """Loss construction."""
        # logits = self.resize(logits)
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_classes))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_classes, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))

        return loss


class SoftmaxCrossEntropyLoss(nn.Cell):
    """SoftmaxCrossEntropyLoss"""

    def __init__(self, num_cls=21, ignore_label=255):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, logits, labels):
        """SoftmaxCrossEntropyLoss.construct"""
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss


class Lovasz_hinge(LossBase):
    '''
    This is the autograd version, used in the multi-category classification case
    '''

    def __init__(self, reduction='mean', num_classes=1, ignore_index=-1):
        super(Lovasz_hinge, self).__init__()
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.num_classes = num_classes
        self.cast = F.Cast()
        self.sort = F.Sort(descending=True)
        self.mul = F.Mul()
        self.print = F.Print()
        self.relu = F.ReLU()

    def construct(self, probs, labels):
        B, C, H, W = probs.shape
        probs = probs.view(-1)
        labels = labels.view(-1)
        labels = self.cast(labels, mstype.float32)
        probs = self.cast(probs, mstype.float32)
        signs = 2. * labels - 1.
        # signs = Parameter()(signs, requires_grad=False)

        errs = (1. - probs * signs)
        errs_sort, errs_order = self.sort(errs)

        gt_sorted = labels[errs_order]
        # lovasz extension grad

        n_samples = gt_sorted.shape[0]
        n_pos = gt_sorted.sum()
        inter = n_pos - gt_sorted.cumsum(0)
        union = n_pos + (1. - gt_sorted).cumsum(0)
        jacc = 1. - inter / union
        if n_samples > 1:
            jacc[1:] = jacc[1:] - jacc[:-1]
        errs_sort = self.relu(errs_sort)
        # jacc = Parameter(jacc)
        losses = self.mul(errs_sort, jacc).sum()
        # self.print(losses)
        return losses