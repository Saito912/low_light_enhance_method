from .uretinex_net import _URetinex_Net
import torch
import cv2
import numpy as np

class URetinex_Net:
    def __init__(self):
        self.model = _URetinex_Net().cuda()
        self.model.eval()

    @torch.no_grad()
    def __call__(self, low_img: np.array):
        low_img = (torch.from_numpy(cv2.cvtColor(low_img,
                                                 cv2.COLOR_BGR2RGB).transpose((2, 0, 1))) / 255.).unsqueeze(0).cuda()
        enhance = self.model(input_low_img=low_img)
        enhance = enhance.cpu().numpy() * 255
        enhance = enhance.transpose(1, 2, 0).astype(np.uint8)
        enhance = cv2.cvtColor(enhance, cv2.COLOR_RGB2BGR)
        return enhance
