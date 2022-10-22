import numpy as np
import torch
import os
from .ruas import Network
import cv2


class RUAS:
    def __init__(self, mode=2, device='cuda'):
        self.device = device
        self.mode = mode
        type_ckpt = ['lol.pt', 'dark.pt', 'upe.pt']
        current_path = os.path.dirname(__file__)
        ckpt_path = os.path.join(current_path, 'ckpt', type_ckpt[mode])
        self.model = Network().to(device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
        self.model.eval()

    @torch.no_grad()
    def __call__(self, low_img):
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        low_img = (torch.from_numpy(low_img).permute(2, 0, 1) / 255.0).float().unsqueeze(0).cuda()
        u_list, t_list = self.model(low_img)
        if self.mode == 0:
            out_img = t_list[-1][0]
        else:
            out_img = u_list[-2][0]
        out_img = out_img.cpu().numpy() * 255
        out_img = out_img.transpose(1, 2, 0).astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        return out_img
