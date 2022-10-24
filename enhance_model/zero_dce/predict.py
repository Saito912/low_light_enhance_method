import numpy as np
import torch
from .zero_dce import _ZeroDCE
import cv2
import os


class ZeroDCE:
    def __init__(self, device='cuda'):
        self.model = _ZeroDCE().to(device)
        ckpt_path = os.path.join(os.path.dirname(__file__), 'zero_dce_ckpt.pth')
        print('load ckpt from ' + ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, low_img):
        low_img = (torch.from_numpy(
            cv2.cvtColor(low_img,
                         cv2.COLOR_BGR2RGB).transpose(2, 0, 1)) / 255.).unsqueeze(0).to(self.device)
        _, enhance, _ = self.model(low_img)
        enhance = enhance[0].cpu().numpy() * 255
        enhance = enhance.transpose(1, 2, 0).astype(np.uint8)
        enhance = cv2.cvtColor(enhance, cv2.COLOR_RGB2BGR)
        return enhance
