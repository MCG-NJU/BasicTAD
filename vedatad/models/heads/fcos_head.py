# import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from vedacore.misc import registry

class ScaleLayer(nn.Module):
    
    def __init__(self, init_value=1.0):
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class DeformConv1D(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2) on 1D Convs.
        """
        super(DeformConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.conv = nn.Conv1d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv1d(inc, kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv1d(inc, kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.cpu().data.type()
        ks = self.kernel_size
        N = offset.size(1)

        if self.padding:
            x = F.pad(x, (1, 1), 'constant', 0)

        # (batchsize, N, T)
        # get two nearest points
        p = self._get_p(offset, dtype)

        # (batchsize, T, N)
        p = p.contiguous().permute(0, 2, 1)
        q_l = p.detach().floor()
        q_r = q_l + 1

        # nearest left point and right point
        q_l = torch.clamp(q_l, 0, x.size(2) - 1)
        q_r = torch.clamp(q_r, 0, x.size(2) - 1)

        # clip p
        p = torch.clamp(p, 0, x.size(2) - 1)

        # get values for left and right points by linear interpolating
        x_q_l = self._get_x_q(x, q_l, N)
        x_q_r = self._get_x_q(x, q_r, N)

        g_l = p - q_l
        g_r = q_r - p

        x_offset = g_r.unsqueeze(dim=1) * x_q_l + g_l.unsqueeze(dim=1) * x_q_r

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype, device):
        p_n = torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1)
        p_n = p_n.view(1, N, 1).type(dtype)
        p_n = p_n.to(device)

        return p_n

    def _get_p_0(self, T, N, dtype, device):
        p_0 = torch.arange(1, T * self.stride + 1, self.stride)
        p_0 = p_0.view(1, 1, T).repeat(1, N, 1).type(dtype)
        p_0 = p_0.to(device)

        return p_0

    def _get_p(self, offset, dtype):
        N, T = offset.size(1), offset.size(2)

        # (1, N, 1, 1)
        p_n = self._get_p_n(N, dtype, offset.device)
        # (1, N, h, w)
        p_0 = self._get_p_0(T, N, dtype, offset.device)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        batchsize, T, __ = q.size()
        channel = x.size(1)
        # (batchsize, channel, T)
        x = x.contiguous().view(batchsize, channel, -1)

        # (batchsize, T, N)
        index = q
        # (batchsize, channel, T*N)
        index = index.long().contiguous().unsqueeze(
            dim=1
        ).expand(-1, channel, -1, -1).contiguous().view(batchsize, channel, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(batchsize, channel, T, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        batchsize, channel, T, N = x_offset.size()
        x_offset = torch.cat([
            x_offset[..., s:s + ks].contiguous().view(batchsize, channel, T * ks)
            for s in range(0, N, ks)
        ], dim=-1)

        return x_offset

@registry.register_module('head')
class FcosHead(nn.Module): 

    def __init__(self, heads, in_channels, num_ins=3,num_blocks=3,kernel_size=3,stride=1,padding=1):
        super(FcosHead, self).__init__()

        self.in_channels = in_channels
        self.num_ins = num_ins
        self.cls_convs = []
        self.reg_convs = []
        
        self.num_blocks=num_blocks         
        self.kernel_size=kernel_size         
        self.stride=stride         
        self.padding=padding

        # add norm convs
        for i in range(3):
            self.cls_convs.append(
                nn.Conv1d(self.in_channels, self.in_channels, self.kernel_size,
                          self.stride, self.padding)
            )
            self.cls_convs.append(nn.ReLU(inplace=True))
            
            self.reg_convs.append(
                nn.Conv1d(self.in_channels, self.in_channels, self.kernel_size,
                          self.stride, self.padding)
            )
            self.reg_convs.append(nn.ReLU(inplace=True))

        # add deformable convs
        self.cls_convs.append(
            DeformConv1D(self.in_channels, self.in_channels,self.kernel_size,self.stride, self.padding,
                         bias=True, modulation=True)
        )
        self.cls_convs.append(nn.ReLU(inplace=True))
        self.reg_convs.append(
            DeformConv1D(self.in_channels, self.in_channels,self.kernel_size,self.stride, self.padding,
                         bias=True, modulation=True)
        )
        self.reg_convs.append(nn.ReLU(inplace=True))

        self.cls_convs = nn.Sequential(*self.cls_convs)
        self.reg_convs = nn.Sequential(*self.reg_convs)

        # cls branch
        self.cls = nn.Conv1d(self.in_channels, heads['act_cls'], kernel_size=3,
                             stride=1, padding=1)
        self.reg = nn.Conv1d(self.in_channels, heads['offset_reg'], kernel_size=3,
                             stride=1, padding=1)
        
        # mulitple scale layer
        self.scales = nn.ModuleList([ScaleLayer(init_value=1.0) for __ in range(self.num_ins)])

        # sigmoid focal loss
        # import pdb; pdb.set_trace()
        if heads['act_cls'] == 20:
            nn.init.normal_(self.cls.weight, std=0.01)
            focal_init_value = -math.log((1 - 0.01) / 0.01)
            self.cls.bias.data.fill_(focal_init_value)

        '''
        print("Frozen all head layers except for roi classification branch")
        for p in self.cls_convs.parameters():
            p.requires_grad = False
        for p in self.reg_convs.parameters():
            p.requires_grad = False
        for p in self.cls.parameters():
            p.requires_grad = False
        for p in self.reg.parameters():
            p.requires_grad = False
        for p in self.scales.parameters():
            p.requires_grad = False
        '''


    def forward(self, feats):
        # import pdb; pdb.set_trace()
        outputs = []
        cls_score_list = []
        reg_offset_list = []

        for i, x in enumerate(feats):
            output = {}
            cls_feat = x
            reg_feat = x

            cls_feat = self.cls_convs(cls_feat)
            reg_feat = self.reg_convs(reg_feat)

            cls_score = self.cls(cls_feat)
            reg_offset = self.reg(reg_feat)
            reg_offset = torch.exp(self.scales[i](reg_offset))

            output['act_cls'] = cls_score
            output['offset_reg'] = reg_offset

            outputs.append(output)

        return outputs