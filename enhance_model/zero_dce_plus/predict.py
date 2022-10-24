import cv2
import numpy as np
import torch

from .model import enhance_net_nopool
import os


class ZeroDceP:
    def __init__(self, device='cuda'):
        current_path = os.path.dirname(__file__)
        ckpt_path = os.path.join(current_path, 'zero_dce_plus.pth')
        print('load ckpt from ' + ckpt_path)
        self.model = enhance_net_nopool(8).to(device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, low_img):
        img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0) / 255.
        img = img.float().to(self.device)
        enhanced_image, params_maps = self.model(img)
        enhanced_image = enhanced_image[0]
        enhanced_image = enhanced_image.cpu().numpy() * 255
        enhanced_image = enhanced_image.transpose(1, 2, 0).astype(np.uint8)
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
        return enhanced_image
