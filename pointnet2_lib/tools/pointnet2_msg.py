import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from lib.config import cfg
from torch.nn.functional import grid_sample
from torch.autograd import Variable
import numpy as np
from torch.nn import MaxPool2d

BatchNorm2d = nn.BatchNorm2d


# ____________CBAM_______

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# _______________________
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    """7*7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=0, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)

        avg_out = self.fc(x)
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print("x",x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        print("x__", x.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, outplanes, stride)
        # self.bn1 = BatchNorm2d(outplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(outplanes, outplanes, 2 * stride)
        self.conv1 = conv7x7(3, 128, 1)
        self.bn1 = BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1=MaxPool2d(kernel_size=2)

        self.conv2 = conv5x5(128, 256, 1)
        self.bn2 = BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = conv3x3(256, 128, 1)
        self.bn3 = BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        #
        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu(out)
        return out


class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

        # self.ca = ChannelAttention(inplanes)
        # self.sa = SpatialAttention()

    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        # out = self.ca(img_features)
        # out = self.sa(img_features)
        # print("out",out.shape)

        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


# ================addition attention (add)=======================#
# class IA_Layer(nn.Module):
#     def __init__(self, channels):
#         print('##############ADDITION ATTENTION(ADD)#########')
#         super(IA_Layer, self).__init__()
#         self.ic, self.pc = channels
#         rc = self.pc // 4
#         self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
#                                    nn.BatchNorm1d(self.pc),
#                                    nn.ReLU())
#         self.conv_img = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
#                                       nn.BatchNorm1d(self.pc),
#                                       nn.ReLU())
#         self.conv_pc = nn.Sequential(nn.Conv1d(self.pc, self.pc, 1),
#                                      nn.BatchNorm1d(self.pc),
#                                      nn.ReLU())
#
#         self.fc_img = nn.Sequential(nn.Linear(self.ic, self.ic // 16),
#                                     nn.ReLU(),
#                                     nn.Linear(self.ic // 16, 1),
#                                     nn.Sigmoid())
#
#         self.fc_pc = nn.Sequential(nn.Linear(self.pc, self.pc // 16),
#                                    nn.ReLU(),
#                                    nn.Linear(self.pc // 16, 1),
#                                    nn.Sigmoid())
#         self.img_out = nn.Sequential(nn.Linear(2, 1),
#                                    nn.Sigmoid())
#
#         self.fc1 = nn.Linear(self.ic, rc)
#         self.fc2 = nn.Linear(self.pc, rc)
#         self.fc3 = nn.Linear(rc, 1)
#
#     def forward(self, img_feas, point_feas):
#         batch = img_feas.size(0)
#         img_feas_f = img_feas.transpose(1, 2).contiguous().view(-1, self.ic)  # BCN->BNC->(BN)C
#         #point_feas_f = point_feas.transpose(1, 2).contiguous().view(-1, self.pc)  # BCN->BNC->(BN)C'
#
#         img_feas_f_att = self.fc_img(img_feas_f)
#         img_feas_f_att = img_feas_f_att.squeeze(1)
#         img_feas_f_att = img_feas_f_att.view(batch, 1, -1)  # B1N
#
#
#         # point_feas_f_att = self.fc_pc(point_feas_f)
#         # point_feas_f_att = point_feas_f_att.squeeze(1)
#         # point_feas_f_att = point_feas_f_att.view(batch, 1, -1)  # B1N
#
#
#         # ri = self.fc1(img_feas_f)
#         # rp = self.fc2(point_feas_f)
#         # att = F.sigmoid(self.fc3(F.tanh(ri + rp))) #BNx1
#
#         # print(img_feas.size(), att.size())
#
#         img_feas_new = self.conv_img(img_feas)
#         #point_feas_new = self.conv_pc(point_feas)
#
#         img_feas_new = img_feas_new * img_feas_f_att
#
#         #point_feas_new = point_feas_new * point_feas_f_att #BCN
#
#
#         avg_out = torch.mean(img_feas_new, dim=2, keepdim=True)
#         max_out, _ = torch.max(img_feas_new, dim=2, keepdim=True)
#         sum_out = torch.cat([avg_out, max_out], dim=2)
#         img_feas_new_att = self.img_out(sum_out)
#         img_feas_new_new = img_feas_new * img_feas_new_att
#
#         return img_feas_new
#
#         # batch = img_feas.size(0)
#         # img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
#         # point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
#         #
#         # img_feas_f_att = self.fc_img(img_feas_f)
#         # img_feas_f = img_feas_f_att * img_feas_f
#         #
#         # point_feas_f_att = self.fc_pc(point_feas_f)
#         # point_feas_f = point_feas_f_att * point_feas_f
#         #
#         #
#         #
#         # ri = self.fc1(img_feas_f)
#         #
#         # rp = self.fc2(point_feas_f)
#         #
#         # att = F.sigmoid(self.fc3(F.tanh(ri + rp))) #BNx1
#         #
#         # att = att.squeeze(1)
#         # att = att.view(batch, 1, -1) #B1N
#         # # print(img_feas.size(), att.size())
#         #
#         # img_feas_new = self.conv1(img_feas)
#         # out = img_feas_new * att
#         # return out


class IA_Layer(nn.Module):
    def __init__(self, channels):
        print('##############ADDITION ATTENTION(ADD)#########')
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                   nn.BatchNorm1d(self.pc),
                                   nn.ReLU())
        self.conv_img = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                      nn.BatchNorm1d(self.pc),
                                      nn.ReLU())
        self.conv_pc = nn.Sequential(nn.Conv1d(self.pc, self.pc, 1),
                                     nn.BatchNorm1d(self.pc),
                                     nn.ReLU())
        self.conv_final = nn.Sequential(nn.Conv1d(self.pc*2, self.pc, 1),
                                   nn.BatchNorm1d(self.pc),
                                   nn.ReLU())

        self.fc_img = nn.Sequential(nn.Linear(self.ic, self.ic // 16),
                                    nn.ReLU(),
                                    nn.Linear(self.ic // 16, 1),
                                    nn.Sigmoid())

        self.fc_pc = nn.Sequential(nn.Linear(self.pc, self.pc // 16),
                                   nn.ReLU(),
                                   nn.Linear(self.pc // 16, 1),
                                   nn.Sigmoid())
        self.img_out = nn.Sequential(nn.Linear(2, 1),
                                   nn.Sigmoid())

        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feas, point_feas):
        img_feas = self.conv_img(img_feas)
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1, 2).contiguous().view(-1, self.pc)  # BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1, 2).contiguous().view(-1, self.pc)  # BCN->BNC->(BN)C'

        ri = self.fc2(img_feas_f)
        img_feas_f_att = F.sigmoid(self.fc3(F.tanh(ri))) #BNx1
        img_feas_f_att = img_feas_f_att.squeeze(1)
        img_feas_f_att = img_feas_f_att.view(batch, 1, -1)  # B1N

        #rp = self.fc2(point_feas_f)
        #point_feas_f_att = F.sigmoid(self.fc3(F.tanh(rp)))  # BNx1
        #point_feas_f_att = point_feas_f_att.squeeze(1)
        #point_feas_f_att = point_feas_f_att.view(batch, 1, -1)  # B1N

        img_feas_f_att_w = img_feas.mul(img_feas_f_att)
        #point_feas_f_att_w = point_feas.mul(point_feas_f_att)

        img_feas_f_att_r = img_feas + img_feas_f_att_w
        #point_feas_f_att_r = point_feas + point_feas_f_att_w

        img_feas_new = self.conv_pc(img_feas_f_att_r)
        #point_feas_new = self.conv_pc(point_feas_f_att_r)

        # ful_mul = torch.mul(img_feas_new,point_feas_new)
        # x_in1   = torch.reshape(img_feas_new,[img_feas_new.shape[0],1,img_feas_new.shape[1],img_feas_new.shape[2]])
        # x_in2   = torch.reshape(point_feas_new,[point_feas_new.shape[0],1,point_feas_new.shape[1],point_feas_new.shape[2]])
        # x_cat   = torch.cat((x_in1, x_in2),dim=1)
        # ful_max = x_cat.max(dim=1)[0]
        # ful_out = torch.cat((ful_mul,ful_max),dim=1)

        #out = self.conv_final(img_feas)


        return img_feas_new


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels=[inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        img_features = self.IA_Layer(img_features, point_features)
        # print("img_features:", img_features.shape)

        # fusion_features = img_features + point_features
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """

    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = grid_sample(feature_map, xy)  # (B,C,1,N)

    return interpolate_feature.squeeze(2)  # (B,C,N)


def get_model(input_channels=6, use_xyz=True):
    return Pointnet2MSG(input_channels=input_channels, use_xyz=use_xyz)


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels=6, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=cfg.RPN.SA_CONFIG.NPOINTS[k],
                    radii=cfg.RPN.SA_CONFIG.RADIUS[k],
                    nsamples=cfg.RPN.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RPN.USE_BN
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        ##################

        # Top layer
        self.toplayer = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        ##################
        if cfg.LI_FUSION.ENABLED:
            self.Img_Block = nn.ModuleList()
            self.Fusion_Conv = nn.ModuleList()
            self.DeConv = nn.ModuleList()
            self.cbam = nn.ModuleList()




            self.conv1 = conv3x3(3, 128, 1)
            self.bn1 = BatchNorm2d(128)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool1 = MaxPool2d(kernel_size=2)

            self.conv2 = conv5x5(128, 256, 1)
            self.bn2 = BatchNorm2d(256)
            self.relu = nn.ReLU(inplace=True)

            self.conv3 = conv3x3(256, 128, 1)
            self.bn3 = BatchNorm2d(128)
            self.relu = nn.ReLU(inplace=True)

            self.cbam = CBAM(128, 16)





            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                # [3, 64, 128, 256, 512]
                self.Img_Block.append(
                    BasicBlock(cfg.LI_FUSION.IMG_CHANNELS[i], cfg.LI_FUSION.IMG_CHANNELS[i + 1], stride=1))

                if cfg.LI_FUSION.ADD_Image_Attention:
                    self.Fusion_Conv.append(
                        Atten_Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.POINT_CHANNELS[i],
                                          cfg.LI_FUSION.POINT_CHANNELS[i]))
                else:
                    # self.cbam.append(CBAM(cfg.LI_FUSION.IMG_CHANNELS[i + 1],16))
                    self.Fusion_Conv.append(
                        Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + cfg.LI_FUSION.POINT_CHANNELS[i],
                                    cfg.LI_FUSION.POINT_CHANNELS[i]))

                self.DeConv.append(nn.ConvTranspose2d(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.DeConv_Reduce[i],
                                                      kernel_size=cfg.LI_FUSION.DeConv_Kernels[i],
                                                      stride=cfg.LI_FUSION.DeConv_Kernels[i]))

            # self.cbam_fusion = CBAM(32,16)
            self.image_fusion_conv = nn.Conv2d(128,
                                               128, kernel_size=1)
            self.image_fusion_bn = torch.nn.BatchNorm2d(128)

            if cfg.LI_FUSION.ADD_Image_Attention:
                self.final_fusion_img_point = Atten_Fusion_Conv(128,
                                                                cfg.LI_FUSION.IMG_FEATURES_CHANNEL,
                                                                cfg.LI_FUSION.IMG_FEATURES_CHANNEL)
            else:
                self.final_fusion_img_point = Fusion_Conv(
                    cfg.LI_FUSION.IMG_FEATURES_CHANNEL + cfg.LI_FUSION.IMG_FEATURES_CHANNEL // 4,
                    cfg.LI_FUSION.IMG_FEATURES_CHANNEL)

        self.FP_modules = nn.ModuleList()

        for k in range(cfg.RPN.FP_MLPS.__len__()):
            pre_channel = cfg.RPN.FP_MLPS[k + 1][-1] if k + 1 < len(cfg.RPN.FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + cfg.RPN.FP_MLPS[k])
            )

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, image=None, xy=None):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]

        if cfg.LI_FUSION.ENABLED:
            #### normalize xy to [-1,1]
            size_range = [1280.0, 384.0]
            xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
            xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0  # = xy / (size_range - 1.) * 2 - 1.
            l_xy_cor = [xy]
            #img = [image]
            #print(image.shape, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            out = self.conv1(image)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool1(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.relu(out)
            out = self.cbam(out)

            #img.append(out)
            #print(out.shape, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_index = self.SA_modules[i](l_xyz[i], l_features[i])


            if cfg.LI_FUSION.ENABLED:
                li_index = li_index.long().unsqueeze(-1).repeat(1, 1, 2)
                li_xy_cor = torch.gather(l_xy_cor[i], 1, li_index)

                # image = self.cbam[i](image)
                #img_gather_feature = Feature_Gather(image, li_xy_cor)  # , scale= 2**(i+1))

                #li_features = self.Fusion_Conv[i](li_features, img_gather_feature)

                l_xy_cor.append(li_xy_cor)


            l_xyz.append(li_xyz)
            l_features.append(li_features)
            """
            torch.Size([2, 3, 384, 1280])
            torch.Size([2, 64, 192, 640])1
            torch.Size([2, 128, 96, 320])2
            torch.Size([2, 256, 48, 160])3
            torch.Size([2, 512, 24, 80])4
            CUDA_VISIBLE_DEVICES=0,1 python train_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml --batch_size 4 --train_mode rcnn_online --epochs 50 --mgpus --ckpt_save_interval 1 --output_dir ./log/Car/full_epnet_iou_branch/   --set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2 RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH True TRAIN.CE_WEIGHT 5.0
            CUDA_VISIBLE_DEVICES=0,1 python train_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml --batch_size 4 --train_mode rcnn_online --epochs 50 --mgpus --ckpt_save_interval 1 --output_dir ./log/Car/full_epnet_without_iou_branch/   --set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2 RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False TRAIN.CE_WEIGHT 5.0


            CUDA_VISIBLE_DEVICES=0,1 python eval_rcnn.py  --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online  --eval_all  --output_dir ./log/Car/full_epnet_without_iou_branch/eval_results/  --ckpt_dir ./log/Car/full_epnet_without_iou_branch/ckpt --set  LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False
            CUDA_VISIBLE_DEVICES=0,1 python eval_rcnn.py  --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online  --eval_all  --output_dir ./log/Car/full_epnet_iou_branch/eval_results/  --ckpt_dir ./log/Car/full_epnet_iou_branch/ckpt --set  LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH True
            CUDA_VISIBLE_DEVICES=0,1 python eval_rcnn.py  --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online  --output_dir ./log/Car/full_epnet_without_iou_branch/eval_results/  --ckpt_dir ./log/Car/full_epnet_without_iou_branch/checkpoint_epoch_45.pth --set  LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False
            """

        # c2 = img[1]
        # c3 = img[2]
        # c4 = img[3]
        # c5 = img[4]
        #
        # p5 = self.toplayer(c5)
        # p4 = self._upsample_add(p5, self.latlayer1(c4))
        # p3 = self._upsample_add(p4, self.latlayer2(c3))
        # p2 = self._upsample_add(p3, self.latlayer3(c2))
        # # Smooth
        # p4 = self.smooth1(p4)
        # p3 = self.smooth2(p3)
        # p2 = self.smooth3(p2)
        #
        # img[1] = p2
        # img[2] = p3
        # img[3] = p4
        # img[4] = p5
        # print(img[1].shape)
        # print(img[2].shape)
        # print(img[3].shape)
        # print(img[4].shape)
        # input()
        # print(p2.shape)
        # print(p3.shape)
        # print(p4.shape)
        # print(p5.shape)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        if cfg.LI_FUSION.ENABLED:
            # for i in range(1,len(img))
            #DeConv = []
            #for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                #DeConv.append(self.DeConv[i](img[i + 1]))
            #de_concat = torch.cat(DeConv, dim=1)

            img_fusion = F.relu(self.image_fusion_bn(self.image_fusion_conv(out)))

            # img_fusion = self.cbam_fusion(img_fusion)
            img_fusion_gather_feature = Feature_Gather(img_fusion, xy)
            l_features[0] = self.final_fusion_img_point(l_features[0], img_fusion_gather_feature)

        return l_xyz[0], l_features[0]


class Pointnet2MSG_returnMiddleStages(Pointnet2MSG):
    def __init__(self, input_channels=6, use_xyz=True):
        super().__init__(input_channels, use_xyz)

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        idxs = []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            idxs.append(idx)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_xyz, l_features, idxs
