import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_interp(x, xp, fp):
    '''
    this fn is similar to np.interp
    :param x:
    :param xp:
    :param fp:
    :return:
    '''


    sort_idx = torch.argsort(xp)
    xp = xp[sort_idx]
    fp = fp[sort_idx]


    right_idxs = torch.searchsorted(xp, x)


    right_idxs = right_idxs.clamp(max=len(xp) - 1)

    left_idxs = (right_idxs - 1).clamp(min=0)

    x_left = xp[left_idxs]
    x_right = xp[right_idxs]
    y_left = fp[left_idxs]
    y_right = fp[right_idxs]


    interp_vals = y_left + ((x - x_left) * (y_right - y_left) / (x_right - x_left))


    interp_vals[x < xp[0]] = fp[0]
    interp_vals[x > xp[-1]] = fp[-1]

    return interp_vals
class VMFV1(nn.Module):
    def __init__(self, dims, kappa_init=5, epsilon=1e-5, train_kappa=True, use_cache=True,min_kappa=0.1):
        '''

        :param dims: 超球体维度
        :param kappa_init:  凝聚度 越大越集中
        :param epsilon: 数值法精度
        :param train_kappa: 是否可训练
        '''
        super().__init__()

        self.kappa = nn.Parameter(torch.tensor(kappa_init,dtype=torch.float32), requires_grad=train_kappa) if train_kappa else kappa_init
        self.train_kappa = train_kappa
        self.dims = dims
        self.epsilon = epsilon
        self.cache_kappa_x = None
        self.cache_kappa_y = None
        self.use_cache = use_cache
        self.min_kappa=min_kappa

    @torch.no_grad()
    def get_kappa_x(self, device):
        if self.use_cache and self.cache_kappa_x is not None:
            if self.cache_kappa_x.device != device:
                self.cache_kappa_x = self.cache_kappa_x.to(device)
            return self.cache_kappa_x
        x = torch.arange(-1 + self.epsilon, 1, self.epsilon, dtype=torch.float64, device=device)

        if self.use_cache:
            self.cache_kappa_x = x
        return x
    def get_kappa(self):
        if not self.train_kappa:
            return self.kappa
        return torch.clamp(self.kappa, min=self.min_kappa)
    def get_kappa_y(self, device):
        '''

        :param device:
        :return:
        '''
        x = self.get_kappa_x(device)
        kappa=self.get_kappa()
        if not self.train_kappa:
            with torch.no_grad():
                if self.use_cache and self.cache_kappa_y is not None:
                    if self.cache_kappa_y.device != device:
                        self.cache_kappa_y = self.cache_kappa_y.to(device)
                    return self.cache_kappa_y, x

                y = kappa * x + torch.log(1 - x ** 2) * (self.dims - 3) / 2
                y = torch.cumsum(torch.exp(y - y.max()), dim=0)

                y = y / y[-1]

                if self.use_cache:
                    self.cache_kappa_y = y
                return y, x
        else:
            y = kappa * x + torch.log(1 - x ** 2) * (self.dims - 3) / 2
            y = torch.cumsum(torch.exp(y - y.max()), dim=0)

            y = y / y[-1]

            return y, x

    def get_pw(self, x):
        noise = torch.rand_like(x)
        dtypes=x.dtype
        device = x.device

        kappa_y,kappa_x=self.get_kappa_y(device)

        pw=torch_interp(noise,kappa_y,kappa_x)
        return pw.to(dtypes)
    
    
    def reparameterize(self,x):
        '''
        重参数化方法来自https://kexue.fm/archives/8404
        :param x: mu  B ... C
        :return:
        '''
        noise=torch.randn_like(x)
        pw=self.get_pw(x)
        nu = noise - torch.sum(noise * x, dim=-1,keepdim=True) * x
        nu = F.normalize(nu, dim=-1, p=2)
        return pw * x + (1 - pw ** 2) ** 0.5 * nu


    def forward(self, x):
        return self.reparameterize(x),self.get_kappa()

class VMFKLLossV1(nn.Module):
    def __init__(self,dims):
        super().__init__()

        self.dims=dims


    def forward(self,kappa):
        kl_loss=kappa/self.dims
        return kl_loss


def iv_torch_complete(v, x):

    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=x.dtype, device=x.device)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    if v.shape != x.shape:
        if v.dim() == 0:
            v = v.expand_as(x)
        elif x.dim() == 0:
            x = x.expand_as(v)

    result = torch.zeros_like(x)


    large_v_mask = (v > 10 * x) & (v > 10)
    if large_v_mask.any():
        vl = v[large_v_mask]
        xl = x[large_v_mask]
        # result[large_v_mask] = (1 / torch.sqrt(2 * torch.pi * vl)) * ((torch.e * xl / (2 * vl)) ** vl)
        log_term = -0.5*torch.log(2*torch.pi*vl) + vl*torch.log(torch.e*xl/(2*vl))
        correction = 1.0 + 1.0/(8*vl) + 9.0/(128*vl**2) + 75.0/(1024*vl**3)
        result[large_v_mask] = torch.exp(log_term) * correction


    mid_mask = ~large_v_mask & (x <= 2 * v)
    if mid_mask.any():
        vm = v[mid_mask]
        xm = x[mid_mask]

        terms = torch.ones_like(xm)
        sum_terms = terms.clone()

        for k in range(1, 30):
            terms = terms * (xm * xm) / (4 * k * (vm + k))
            sum_terms = sum_terms + terms
            if torch.all(torch.abs(terms) < 1e-10 * torch.abs(sum_terms)):
                break

        result[mid_mask] = sum_terms * (xm / 2) ** vm / torch.exp(torch.lgamma(vm + 1))

    large_x_mask = ~large_v_mask & ~mid_mask
    if large_x_mask.any():
        vx = v[large_x_mask]
        xx = x[large_x_mask]


        result[large_x_mask] = (1 / torch.sqrt(2 * torch.pi * xx)) * torch.exp(xx)


        v_squared = vx * vx
        correction = 1.0 - (4 * v_squared - 1) / (8 * xx)
        result[large_x_mask] = result[large_x_mask] * correction

    return result


def ive_torch_complete(v, x):

    return iv_torch_complete(v, x) * torch.exp(-torch.abs(x))

def _vmfKL_t(k, d):
    return k * ((iv_torch_complete(d / 2.0 + 1.0, k)+ iv_torch_complete(d / 2.0, k) * d / (2.0 * k)) / iv_torch_complete(d / 2.0, k) - d / (2.0 * k)) + d * torch.log(k) / 2.0 - torch.log(iv_torch_complete(d / 2.0, k)) - torch.lgamma(d / 2 + 1) - d * np.log(2) / 2


class VMFKLLossV2(nn.Module):
    def __init__(self,dims):
        super().__init__()

        self.dims=dims


    def forward(self,kappa):

        return _vmfKL_t(kappa, self.dims)

if __name__ == '__main__':
    import scipy
    m = VMFV1(32)
    x = torch.randn(4,20, 32)
    print(m(x))













