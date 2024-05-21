import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class fc(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.conv=nn.Conv1d(ch,ch,kernel_size=1,padding=0,groups=ch)

    def forward(self,x):
        return self.conv(x)


class tgmc(nn.Module):
    def __init__(self,ch,k):
        super().__init__()
        assert ch%2==0
        self.conv=nn.Conv1d(ch,ch//2,kernel_size=k,padding=(k)//2)
        self.fc=fc(ch//2)

    def forward(self,x):
        x1=self.conv(x)
        x2=self.fc(x1)
        return torch.cat([x1,x2],dim=1)

class tgmm(nn.Module):
    def __init__(self,ch,k,lays):
        super().__init__()
        assert ch%2==0
        self.M=nn.ModuleList([nn.Sequential(tgmc(ch,k),nn.ReLU(),nn.BatchNorm1d(ch),tgmc(ch,k),nn.BatchNorm1d(ch)) for _ in range(lays)])

    def forward(self,x):
        for m in self.M:
            x=m(x)+x
        return x

if __name__ == '__main__':
    m=tgmm(64,3,3)
    x=torch.randn(1, 64, 128)
    xout=m(x)
    print(xout.shape)
