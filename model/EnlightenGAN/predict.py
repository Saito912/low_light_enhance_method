import cv2
import torch
from .enlightengan import _EnlightenGAN
import os
import numpy as np

class EnlightenGAN:
    def __init__(self, device='cuda'):
        self.model = _EnlightenGAN().to(device)
        ckpt_path = os.path.join(os.path.dirname(__file__),'EnlightenGAN.pth')
        print('load ckpt from '+ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path,map_location=device))
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self,low_img:np.array):
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        low_img = (torch.from_numpy(low_img).permute(2, 0, 1) / 255.0).float().to(self.device)
        low_img = (low_img - 0.5) / 0.5
        r, g, b = low_img[0] + 1, low_img[1] + 1, low_img[2] + 1
        A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        A_gray = A_gray.unsqueeze(0).unsqueeze(0)
        low_img = low_img.unsqueeze(0)
        out_img = self.model(low_img, A_gray)[0]
        out_img = out_img * 0.5 + 0.5
        out_img = out_img.cpu().numpy() * 255
        out_img = out_img.transpose(1, 2, 0).astype(np.uint8)
        out_img = np.maximum(out_img, 0)
        out_img = np.minimum(out_img, 255)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        return out_img

