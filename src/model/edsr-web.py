import torch.nn as nn
from .common import *

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


class EDSR(nn.Module):
    """
    modified from https://github.com/thstkdgus35/EDSR-PyTorch
        parser.add_argument('--scale', type=str, default='4',
                        help='super resolution scale')
        parser.add_argument('--rgb_range', type=int, default=255,
                            help='maximum value of RGB')
        parser.add_argument('--n_colors', type=int, default=3,
                            help='number of color channels to use'
        parser.add_argument('--n_resblocks', type=int, default=16,
                            help='number of residual blocks')
        parser.add_argument('--n_feats', type=int, default=64,
                            help='number of feature maps')
    """
    def __init__(self, in_ch, ou_ch, upscale_factor):
        super(EDSR, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3 
        scale = upscale_factor
        conv = default_conv
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(in_ch, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1.0
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, ou_ch, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x - 0.5
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = x + 0.5
        return x 