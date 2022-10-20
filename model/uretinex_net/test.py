import argparse
import torch.nn as nn
from .network.Math_Module import P, Q
from .network.decom import Decom
import torchvision.transforms as transforms
from .utils import *
import cv2
import numpy as np

def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)

class URetinex_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_opt()
        # loading decomposition model 
        self.model_Decom_low = Decom()
        self.model_Decom_low = load_initialize(self.model_Decom_low, self.opts.Decom_model_low_path)
        # loading R; old_model_opts; and L model
        self.unfolding_opts, self.model_R, self.model_L= load_unfolding(self.opts.unfolding_model_path)
        # loading adjustment model
        self.adjust_model = load_adjustment(self.opts.adjust_model_path)
        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
        ]
        self.transform = transforms.Compose(transform)
        self.eval()

    def init_opt(self):
        parser = argparse.ArgumentParser(description='Configure')
        parser.add_argument('--ratio', type=int, default=5)
        parser.add_argument('--Decom_model_low_path', type=str, default="model/uretinex_net/ckpt/init_low.pth")
        parser.add_argument('--unfolding_model_path', type=str, default="model/uretinex_net/ckpt/unfolding.pth")
        parser.add_argument('--adjust_model_path', type=str, default="model/uretinex_net/ckpt/L_adjust.pth")
        parser.add_argument('--gpu_id', type=int, default=0)
        self.opts = parser.parse_args()

    def unfolding(self, input_low_img):
        for t in range(self.unfolding_opts.round):      
            if t == 0: # initialize R0, L0
                P, Q = self.model_Decom_low(input_low_img)
            else: # update P and Q
                w_p = (self.unfolding_opts.gamma + self.unfolding_opts.Roffset * t)
                w_q = (self.unfolding_opts.lamda + self.unfolding_opts.Loffset * t)
                P = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q = self.Q(I=input_low_img, P=P, L=L, lamda=w_q) 
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
        return R, L
    
    def lllumination_adjust(self, L, ratio):
        ratio = torch.ones(L.shape).cuda() * self.opts.ratio
        return self.adjust_model(l=L, alpha=ratio)
    
    def forward(self, input_low_img):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
        with torch.no_grad():
            R, L = self.unfolding(input_low_img)
            High_L = self.lllumination_adjust(L, self.opts.ratio)
            I_enhance = High_L * R
        return I_enhance[0]

    def run(self, low_img:np.array):
        low_img = (torch.from_numpy(cv2.cvtColor(low_img,
                                                 cv2.COLOR_BGR2RGB).transpose((2, 0, 1)))/255.).unsqueeze(0).cuda()
        enhance = self.forward(input_low_img=low_img)
        enhance = enhance.cpu().numpy()*255
        enhance = enhance.transpose(1,2,0).astype(np.uint8)
        enhance = cv2.cvtColor(enhance, cv2.COLOR_RGB2BGR)
        return enhance


