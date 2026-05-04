import torch
from copy import deepcopy

class EMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.ema_model = deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema_model.state_dict()
        for k in esd.keys():
            esd[k].mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()