import torch
import torch.nn as nn
from models.torchutils import boundary_2d, auto_interpolate_2d

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.CE_loss = nn.CrossEntropyLoss(
            reduction=reduction, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class BCELoss(nn.Module):
    def __init__(self, reduction="mean", pos_weight=1.0):
        pos_weight = torch.tensor(pos_weight).cuda()
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=pos_weight)

    def forward(self, prediction, targets):
        return self.bce_loss(prediction, targets.float())
    
    
class BCELoss_soft(nn.Module):
    def __init__(self, reduction="mean", pos_weight=1.0):
        pos_weight = torch.tensor(pos_weight).cuda()
        super(BCELoss_soft, self).__init__()
        self.bce_loss = nn.CrossEntropyLoss(
            reduction=reduction, label_smoothing=0.1)

    def forward(self, prediction, targets):
        return self.bce_loss(prediction, targets)


class CELoss(nn.Module):
    def __init__(self, weight=[1, 1], ignore_index=-100, reduction='mean'):
        super(CELoss, self).__init__()
        weight = torch.tensor(weight).cuda()
        self.CE = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target.squeeze(1).long())
        return loss


class CELoss_new(nn.Module):
    def __init__(self, weight=[1, 1], ignore_index=-100, reduction='mean'):
        super(CELoss_new, self).__init__()
        weight = torch.tensor(weight).cuda()
        self.CE = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target.long())
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        intersection = 2 * torch.sum(prediction * target) + self.smooth
        union = torch.sum(prediction) + torch.sum(target) + self.smooth
        loss = 1 - intersection / union
        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self, reduction="mean", D_weight=0.5):
        super(CE_DiceLoss, self).__init__()
        self.DiceLoss = DiceLoss()
        self.BCELoss = BCELoss(reduction=reduction)
        self.D_weight = D_weight

    def forward(self, prediction, targets):
        return self.D_weight * self.DiceLoss(prediction, targets) + (1 - self.D_weight) * self.BCELoss(prediction,
                                                                                                       targets)







class Interpolate2dLoss(nn.Module):
    """
    This function adds interpolation on input x or y before the 
    loss computation. This function uses the auto_interpolate_2d that 
    resizes tensor automatically based on its type.
    """

    def __init__(self, lossf, resize_x=False, align_corners=True):
        """
        Args:
            loss_fn: a torch.nn specified loss function. Reduction has 
            to be "none". Or an element-wise operation on two same size 
            torch.Tensor.
            resize_x: a bool. When True, x is resized to y (ground truth)
            size, vice versa. 
            align_corners: bool, whether to align the corners.
        """
        super().__init__()
        self.lossf = lossf
        self.resize_x = resize_x
        self.align_corners = align_corners

    def forward(self, x, y):
        """
        Args:
            x: [bs x c x 2D] float32 tensor, the prediction. 
            y: [bs x c x 2D] float32 tensor or [bs x 2D] int64 tensor, 
            ground truth (label) tensor. 
        Returns:
            l: [bs x c x 2D] or [bs x 2D] float32 tensor, 
            the non-reduced loss. 
        """
        if self.resize_x:
            x = auto_interpolate_2d(
                size=y.shape[-2:], 
                align_corners=self.align_corners)(x)
        else:
            y = auto_interpolate_2d(
                size=x.shape[-2:], 
                align_corners=self.align_corners)(y)
        l = self.loss_fn(x, y)
        return l


class BoundaryCELoss(Interpolate2dLoss):
    """
    Loss function that ignore all nonboundary part.
    """
    def __init__(self, 
                 ignore_label=None,
                 resize_x=False,
                 align_corners=True
                 ):
        if ignore_label is None:
            ignore_label = -100
        super().__init__(
            lossf=nn.CrossEntropyLoss(ignore_index=ignore_label),
            resize_x=resize_x,
            align_corners= align_corners
        )
        
        self.boundary_f = boundary_2d(ignore_label=ignore_label)
        self.ignore_label = ignore_label

    def forward(self, x, y):
        """
        Args:
            x: [bs x c x 2D] float32 tensor, 
                the prediction. -> prediction인데 지금 Logit이 들어오게 됨
            y: [bs x c x 2D] float32 tensor or [bs x 2D] int64 tensor, 
                ground truth (label) tensor. 
        Returns:
            l: [bs x c x 2D] or [bs x 2D] float32 tensor, 
                the non reduced loss. 
        """
        if self.resize_x:
            x = auto_interpolate_2d(
                size=y.shape[-2:], 
                align_corners=self.align_corners)(x)
        else:
            y = auto_interpolate_2d(
                size=x.shape[-2:], 
                align_corners=self.align_corners)(y)


        # if it is ignore label -> we don't count it as boundary
        m = self.boundary_f(y)!=1
        y[m] = self.ignore_label
        l = self.lossf(x, y)
        return l






class Plus_Boundary_Loss(nn.Module):
    def __init__(self, reduction="mean", D_weight=0.1):
        super(Plus_Boundary_Loss, self).__init__()
        self.BoundaryCELoss = BoundaryCELoss(ignore_label = -1)
        self.BCELoss = BCELoss(reduction=reduction)
        self.D_weight = D_weight

    def forward(self, prediction, targets):
        return self.D_weight * self.BoundaryCELoss(prediction, targets) + (1 - self.D_weight) * self.BCELoss(prediction,
                                                                                                       targets)