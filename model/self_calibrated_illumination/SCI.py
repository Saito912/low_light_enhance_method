import os

import cv2
import numpy as np
import torch
import torch.utils

from .model import Finetunemodel


class SCI:
    def __init__(self):
        current_work_dir = os.path.dirname(__file__)
        print('load ckpt from '+os.path.join(current_work_dir,'weights/difficult.pt'))
        model_weight_path = os.path.join(current_work_dir,'weights/difficult.pt')
        self.model = Finetunemodel(model_weight_path).cuda()
        self.model.eval()

    @torch.no_grad()
    def run(self,low_img):
        low_img = (torch.from_numpy(cv2.cvtColor(low_img,
                                                 cv2.COLOR_BGR2RGB).transpose((2, 0, 1))) / 255.).unsqueeze(0).cuda()
        i, enhance = self.model(low_img)
        enhance = enhance[0]
        enhance = enhance.cpu().numpy() * 255
        enhance = enhance.transpose(1, 2, 0).astype(np.uint8)
        enhance = cv2.cvtColor(enhance, cv2.COLOR_RGB2BGR)
        return enhance

