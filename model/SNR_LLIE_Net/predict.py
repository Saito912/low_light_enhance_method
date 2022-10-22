from .low_light_transformer import low_light_transformer
import os
import torch
import cv2
import numpy as np

class SNR:
    def __init__(self,mode=0):
        self.model = low_light_transformer(64,5,8,1,1,predeblur=True,HR_in=True,w_TSA=True).cuda()
        current_path = os.path.dirname(__file__)
        type_ckpt = ['indoor_G.pth','LOLv1.pth','LOLv2_real.pth','LOLv2_synthetic.pth','outdoor_G.pth','SID.pth','SMID.pth']
        assert mode < len(type_ckpt), f'mode must less than len(type_ckpt)={len(type_ckpt)}, current mode is {mode}'
        ckpt_path = os.path.join(current_path,'weights',type_ckpt[mode])
        print('load ckpt from '+ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path,map_location='cuda'))

        self.model.eval()

    @torch.no_grad()
    def __call__(self,low_img:np.array):
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        img_nf = low_img.astype(np.float32)
        img_nf = cv2.blur(img_nf, (5, 5))
        img_nf = img_nf / 255.0
        img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1).unsqueeze(0).cuda()
        low_img = (torch.from_numpy(low_img.transpose((2, 0, 1))) / 255.).unsqueeze(0).cuda()

        dark = low_img
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = img_nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)
        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()

        enhance = self.model(low_img,mask)
        enhance = enhance[0]
        enhance = enhance.cpu().numpy() * 255
        enhance = enhance.transpose(1, 2, 0).astype(np.uint8)
        enhance = cv2.cvtColor(enhance, cv2.COLOR_RGB2BGR)
        return enhance