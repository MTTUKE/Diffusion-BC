import torch

def _flatten(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x[:, 0]
    return x.reshape(x.shape[0], -1)

@torch.no_grad()
def dice_iou(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-7):

    p = _flatten(pred_bin.float())
    t = _flatten(target_bin.float())

    tp = (p * t).sum(dim=1)
    fp = (p * (1 - t)).sum(dim=1)
    fn = ((1 - p) * t).sum(dim=1)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return dice.mean().item(), iou.mean().item()
