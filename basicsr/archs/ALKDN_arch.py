import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights


class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()

        self.dw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.pw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, input):
        out = self.dw(input)
        out = self.pw(out)
        return out


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        padding = kernel_size // 2
        self.dw = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


class ALKConv(nn.Module):
    """
    we refer to https://github.com/VITA-Group/SLaK to implement our designed Asymmetric Large Kernel Convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, small_size=1, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.LoRA1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(kernel_size, small_size),
                               stride=stride, padding=(kernel_size//2, small_size//2), dilation=1, groups=out_channels)
        self.LoRA2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(small_size, kernel_size),
                               stride=stride, padding=(small_size//2, kernel_size//2), dilation=1, groups=out_channels)
        self.act = nn.GELU()
        self.small_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                    stride=stride, padding=1, dilation=1, groups=out_channels)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.act(self.LoRA1(fea)) + self.act(self.LoRA2(fea)) + self.act(self.small_conv(fea))
        fea = fea.permute(0, 2, 3, 1)  # (B, H, W, C)
        fea = self.norm(fea)
        fea = fea.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return fea


def stdv_channels(F):
    # assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    # assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        conv = BSConvU
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        c1_ = (self.conv1(input))
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3 + cf))
        m = self.sigmoid(c4)

        return input * m


class LKDB(nn.Module):
    """
        code based on: [github] https://github.com/xiaom233/BSRN
        we add layer norm to improve the stability of the training process
    """
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, p=0.25, lk_size=3):
        super(LKDB, self).__init__()
        kwargs = {'padding': 1}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=lk_size, **kwargs)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=lk_size, **kwargs)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=lk_size, **kwargs)

        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=lk_size, **kwargs)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, conv)
        self.cca = CCALayer(in_channels)
        self.pixel_norm = nn.LayerNorm(out_channels)  # channel-wise
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        out_fused = self.esa(out)
        out_fused = self.cca(out_fused)
        out_fused = out_fused.permute(0, 2, 3, 1)  # (B, H, W, C)
        out_fused = self.pixel_norm(out_fused)
        out_fused = out_fused.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return out_fused + input


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


@ARCH_REGISTRY.register()
class ALKDN(nn.Module):
    """
        code based on: [github] https://github.com/xiaom233/BSRN
        we modified the upsampler part to apply ABRL
    """
    def __init__(self, num_in_ch=3, num_feat=64, num_block=8, num_out_ch=3, upscale=4,
                 conv='LargeKernelConv', p=0.25, lk_size=3):
        super(ALKDN, self).__init__()
        kwargs = {'padding': 1}
        # print(conv)
        if conv == 'DepthWiseConv':
            self.conv = DepthWiseConv
        elif conv == 'BSConvU':
            self.conv = BSConvU
        elif conv == 'LargeKernelConv':
            self.conv = ALKConv
        else:
            self.conv = nn.Conv2d
        self.fea_conv = BSConvU(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)
        self.upscale = upscale

        self.B1 = LKDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p, lk_size=lk_size)
        self.B2 = LKDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p, lk_size=lk_size)
        self.B3 = LKDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p, lk_size=lk_size)
        self.B4 = LKDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p, lk_size=lk_size)
        self.B5 = LKDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p, lk_size=lk_size)
        self.B6 = LKDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p, lk_size=lk_size)
        self.B7 = LKDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p, lk_size=lk_size)
        self.B8 = LKDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p, lk_size=lk_size)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)
        self.c_up = nn.Conv2d(num_feat, (upscale ** 2) * num_out_ch, 3, 1, 1)
        self.to_img = nn.PixelShuffle(upscale)

    def forward(self, input):
        x = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(x)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)

        out_lr = self.c2(out_B) + out_fea

        output = self.to_img(self.c_up(out_lr) + torch.repeat_interleave(input, repeats=self.upscale ** 2, dim=1))

        return output
