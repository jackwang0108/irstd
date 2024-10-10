import torch
import torch.nn as nn
# from einops import rearrange, repeat
import math
import torch.nn.functional as F

import numpy as np

__all__ = ['RPCANetHQS']

class RPCANetHQS(nn.Module):
    def __init__(self, stage_num=6, slayers=6, channel=32, mode='train'):
        super(RPCANetHQS, self).__init__()
        self.stage_num = stage_num
        self.decos = nn.ModuleList()
        self.mode = mode
        for _ in range(stage_num):
            self.decos.append(DecompositionModule(slayers=slayers, channel=channel))
            
        self.hy = HyPaNet(in_nc=2, out_nc=stage_num*3, channel=channel)
        # '''
        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # '''
        # print("------------------------", self.decos)
    def forward(self, D):
        # T = torch.zeros(D.shape).to(D.device)
        # Z = torch.randn(D.shape).to(D.device)        
        T = torch.zeros_like(D)
        Z = torch.randn_like(D)
        
        mu, lamd, gama = self.hy(torch.cat((D, torch.tensor([1.0]).type_as(D).expand_as(D)), dim=1))
        
        for i in range(self.stage_num):
            D, T = self.decos[i](D, T, Z, mu[:,[i],:,:], lamd[:,[i],:,:], gama[:,[i],:,:])
        if self.mode == 'train':
            return D,T
        else:
            return T

# class DecompositionModule(object):
#     pass


class DecompositionModule(nn.Module):
    def __init__(self, slayers=6, channel=32):
        super(DecompositionModule, self).__init__()
        # self.lowrank = LowrankModule(channel=channel, layers=llayers)
        # self.sparse = SparseModule(channel=channel, layers=slayers)
        # self.merge = MergeModule(channel=channel, layers=mlayers)
        
        self.lowrank = LowrankModule()
        self.znet = ZnetModule(channel=channel, layers=slayers)
        self.sparse = SparseModule()
        

    def forward(self, D, T, Z, mu,lamd,gama):
        # B = self.lowrank(D, T)
        # T = self.sparse(D, B, T)
        # D = self.merge(B, T)        
        B = self.lowrank(D, T, Z,mu)
        Z = self.znet(B, lamd)  #Z = self.znet(B, self.lowrank.mu,lamd)
        T = self.sparse(D, B, gama)
        
        return D, T


class LowrankModule(nn.Module):
    def __init__(self):
        super(LowrankModule, self).__init__()        
        # self.mu = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, D, T, Z, mu):       
        x = D - T + mu*Z
        B = x / (1+mu)
        return B

class ZnetModule(nn.Module):
    def __init__(self, channel=32, layers=6):
        super(ZnetModule, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        
        # self.lamd = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.mlp = nn.Sequential(
            nn.Conv2d(1, 64, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1, 1, 0),
        )        

    def forward(self, B, lamd):
        # x = torch.cat((x, ab[:, (i+self.n):(i+self.n+1), ...].repeat(1, 1, x.size(2), x.size(3))), dim=1)
        # param = lamd / (mu+1e-6)
        param_feature = self.mlp(lamd)
        # x = B + lamd #B + param  
        x = B + param_feature      
        Z = self.convs(x)
        
        return Z

class SparseModule(nn.Module):
    def __init__(self):
        super(SparseModule, self).__init__()
        # self.gama = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, D, B, gama):            
        x = D - B
        m = torch.tensor([0.0]).to(D.device)
        param = torch.max(gama, m)  #.to(D.device)
        # T = torch.sign(x) * torch.max(torch.abs(x) - param, torch.tensor([0.0])) 
        T = torch.max(m, x-param) + torch.min(m, x+param)        
        T = torch.max(T, m)
        
        return T


class HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())

    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x.chunk(chunks=3,dim=1)


'''
class LowrankModule(nn.Module):
    def __init__(self):
        super(LowrankModule, self).__init__()        
        self.mu = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, D, T, Z):       
        x = D - T + self.mu*Z
        B = x / (1+self.mu)
        return B

class ZnetModule(nn.Module):
    def __init__(self, channel=32, layers=6):
        super(ZnetModule, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        
        self.lamd = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.mlp = nn.Sequential(
            nn.Conv2d(1, 64, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1, 1, 0),
        )        

    def forward(self, B, mu):
        # x = torch.cat((x, ab[:, (i+self.n):(i+self.n+1), ...].repeat(1, 1, x.size(2), x.size(3))), dim=1)
        param = self.lamd / (mu+1e-6)
        # param_feature = self.mlp(param.unsqueeze(0).unsqueeze(0).unsqueeze(0))
        x = B + param        
        Z = self.convs(x)
        
        return Z

class SparseModule(nn.Module):
    def __init__(self):
        super(SparseModule, self).__init__()
        self.gama = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, D, B):            
        x = D - B
        m = torch.tensor([0.0]).to(D.device)
        param = torch.max(self.gama, m)  #.to(D.device)
        # T = torch.sign(x) * torch.max(torch.abs(x) - param, torch.tensor([0.0])) 
        T = torch.max(m, x-param) + torch.min(m, x+param)        
        T = torch.max(T, m)
        
        return T
'''

'''
class LowrankModule(nn.Module):
    def __init__(self, channel=32, layers=3):
        super(LowrankModule, self).__init__()

        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        #self.relu = nn.ReLU()
        #self.gamma = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

    def forward(self, D, T):
        x = D - T
        B = x + self.convs(x)
        return B


class SparseModule(nn.Module):
    def __init__(self, channel=32, layers=6) -> object:
        super(SparseModule, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        self.epsilon = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

    def forward(self, D, B, T):
        # print(D.shape)
        x = T + D - B
        T = x - self.epsilon * self.convs(x)
        return T


class MergeModule(nn.Module):
    def __init__(self, channel=32, layers=3):
        super(MergeModule, self).__init__()
        convs = [nn.Conv2d(1, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.mapping = nn.Sequential(*convs)

    def forward(self, B, T):
        x = B + T
        D = self.mapping(x)
        return D
'''