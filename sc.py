import torch
import torch.nn as nn
import torch.nn.functional as F




class EncBlock(nn.Module):
    def __init__(self, indim, outdim, down_factor, time_emb_dim):
        super().__init__()
        self.down_conv = nn.Conv1d(in_channels=indim, out_channels=outdim, kernel_size=down_factor * 2,
                                   stride=down_factor, padding=(down_factor + 1) // 2)
        self.glu = nn.GLU(dim=1)
        self.conv1 = nn.Conv1d(in_channels=outdim, out_channels=outdim, kernel_size=15, padding=15 // 2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=outdim, out_channels=outdim * 2, kernel_size=3, padding=1)
        self.time_emb = nn.Linear(time_emb_dim, outdim)

    def forward(self, x, t):
        x = self.down_conv(x)
        x = self.act(x)
        x = x + self.time_emb(t).transpose(1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.glu(x)
        return x


class DecBlock(nn.Module):
    def __init__(self, indim, outdim, up_factor, time_emb_dim):
        super().__init__()
        self.mix_conv = nn.Conv1d(in_channels=indim * 2, out_channels=indim, kernel_size=3, padding=1)
        self.up_conv = nn.ConvTranspose1d(in_channels=indim, out_channels=outdim, kernel_size=up_factor * 2,
                                          stride=up_factor, padding=(up_factor + 1) // 2)
        self.glu = nn.GLU(dim=1)
        self.conv1 = nn.Conv1d(in_channels=outdim, out_channels=outdim, kernel_size=15, padding=15 // 2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=outdim, out_channels=outdim * 2, kernel_size=5, padding=2)
        self.time_emb = nn.Linear(time_emb_dim, outdim)

    def forward(self, x, res, t):
        x = torch.cat([x, res], dim=1)
        x = self.mix_conv(x)
        x = self.act(x)
        x = self.up_conv(x)
        x = self.act(x)
        x = x + self.time_emb(t).transpose(1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.glu(x)
        return x


class UnetDBlock(nn.Module):
    def __init__(self, indim=1, dim=[16, 32, 64, 128, 256, 512], down_factor=[2, 2, 2, 2, 2], time_emb_dim=256):
        super().__init__()
        assert (len(dim) - 1) == len(down_factor)
        self.pre_conv = nn.Conv1d(in_channels=indim, out_channels=dim[0], kernel_size=7, padding=7 // 2)
        self.enc_blocks = nn.ModuleList()

        for idx, d in enumerate(down_factor):
            self.enc_blocks.append(
                EncBlock(indim=dim[idx], outdim=dim[idx + 1], down_factor=d, time_emb_dim=time_emb_dim))

    def forward(self, x, t):
        res = []
        x = self.pre_conv(x)
        x1 = x
        for block in self.enc_blocks:
            x = block(x, t)
            res.append(x)
        res.reverse()
        return x, res, x1


class UnetUBlock(nn.Module):
    def __init__(self, indim=1, dim=[16, 32, 64, 128, 256, 512], up_factor=[2, 2, 2, 2, 2], time_emb_dim=256):
        super().__init__()
        self.out_conv = nn.Conv1d(in_channels=dim[0]*2 , out_channels=indim, kernel_size=7, padding=7 // 2)
        self.dec_blocks = nn.ModuleList()
        up_factor.reverse()
        dim.reverse()

        for idx, d in enumerate(up_factor):
            self.dec_blocks.append(
                DecBlock(indim=dim[idx], outdim=dim[idx + 1], up_factor=d, time_emb_dim=time_emb_dim))

    def forward(self, x, res, x1, t):

        for idx, block in enumerate(self.dec_blocks):
            x = block(x, res[idx], t)
        x = torch.cat([x, x1], dim=1)
        # x=x+x1
        x = self.out_conv(x)
        return x
class UnetUBlockV2(nn.Module):
    def __init__(self, indim=1, dim=[16, 32, 64, 128, 256, 512], up_factor=[2, 2, 2, 2, 2], time_emb_dim=256):
        super().__init__()
        self.out_conv = nn.Conv1d(in_channels=dim[0] , out_channels=indim, kernel_size=7, padding=7 // 2)
        self.dec_blocks = nn.ModuleList()
        up_factor.reverse()
        dim.reverse()

        for idx, d in enumerate(up_factor):
            self.dec_blocks.append(
                DecBlock(indim=dim[idx], outdim=dim[idx + 1], up_factor=d, time_emb_dim=time_emb_dim))

    def forward(self, x, res, x1, t,cond):

        for idx, block in enumerate(self.dec_blocks):
            x = block(x, res[idx]+cond[idx], t)
        # x = torch.cat([x, x1], dim=1)
        x = self.out_conv(x)
        return x

class cresBlock(nn.Module):
    def __init__(self, dim=16,lays=3):
        super().__init__()
        self.enc_blocks = nn.ModuleList()

        for _ in range(lays):
            self.enc_blocks.append(nn.Sequential(
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=11, padding=11 // 2),
                nn.GELU(),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=11, padding=11 // 2),
                nn.GELU(),
            ))
    def forward(self, x):
        for block in self.enc_blocks:
            x = block(x)+x
        return x


class GCUUnetBlock(nn.Module):
    def __init__(self, indim=1,
                 dim=[16, 32, 64, 128, 256, 512],
                 down_factor=[2, 2, 2, 2, 2],
                 num_heads=8,
                 query_key_dim=256,
                 lays=2,
                 expansion_factor=2,
                 conv_norm_type="RMS",
                 convnext_block_drop=0.1,
                 conv_bias=True,
                 kernel_size=31,
                 attention_out_drop=0.1,
                 model_norm_fn_type='RMS',
                 model_norm_type='pre',
                 time_emb_dim=256):
        super().__init__()
        self.unet_d_block = UnetDBlock(indim=indim, dim=dim, down_factor=down_factor,time_emb_dim=time_emb_dim)
        self.unet_u_block = UnetUBlock(indim=indim, dim=dim.copy(), up_factor=down_factor.copy(),time_emb_dim=time_emb_dim)


        # self.res = cresBlock(dim[0])

    def forward(self, x, t):
        x, res, x1 = self.unet_d_block(x, t)
        # x1=self.res(x1)

        x = self.unet_u_block(x, res, x1, t)
        return x
