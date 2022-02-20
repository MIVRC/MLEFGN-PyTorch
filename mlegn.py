from model import common
import torch
import torch.nn as nn
from torch.nn import  Parameter, Softmax
import torch.nn.functional as F

def make_model(args, parent=False):
    return MLEGN(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kernel_size = 3):
        super(RDB_Conv, self).__init__()
        n_feats = inChannels
        rate  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(n_feats, rate, kernel_size, padding=(kernel_size-1)//2, stride=1),
            nn.ReLU()
        ])
    
    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class FEB(nn.Module):
    def __init__(self, args, n_layer):
        super(FEB, self).__init__()
        n_feats = args.n_feats
        rate  = args.rate
        kernel_size = args.kernel_size
        
        convs = []
        for n in range(n_layer):
            convs.append(RDB_Conv(n_feats + n * rate, rate))
        self.convs = nn.Sequential(*convs)
        
        self.LFF = nn.Conv2d(n_feats + n_layer * rate, n_feats, 1, padding=0, stride=1)
    
    def forward(self, x):
        out = self.LFF(self.convs(x)) + x
        return out

class EA_Module(nn.Module):
    """ Edge Attention Module"""
    def __init__(self, args):
        super(EA_Module, self).__init__()
        in_dim = args.in_dim
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        
        self.softmax = Softmax(dim=-1)
    def forward(self, x, edge):
        """
            inputs :
            x    : input image feature maps( B X C X H X W)
            edge : input edge maps( B X C X H X W)
            returns :
            out : attention value + input feature
            attention: B X (HxW) X (HxW)
            """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(edge).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        
        out = self.gamma*out + x
        return out

class RG(nn.Module):
    """ Residual Group"""
    def __init__(self, n_feats, block, kernel_size=3, conv=common.default_conv):
        super(RG, self).__init__()
        n_resblock = block
        
        residual_group = []
        residual_group = [
            common.ResBlock(
                conv, n_feats, kernel_size
                ) for _ in range(n_resblock)]
        self.body = nn.Sequential(*residual_group)

    def forward(self, x):
        res = self.body(x)
        out = res + x
        return out

class Edge_Net(nn.Module):
    def __init__(self, args, n_feats, n_layer, kernel_size=3, conv=common.default_conv):
        super(Edge_Net, self).__init__()

        self.trans = conv(args.n_colors, n_feats, kernel_size)
        self.head = common.ResBlock(conv, n_feats, kernel_size)
        self.rdb = FEB(args, n_layer)
        self.tail = common.ResBlock(conv, n_feats, kernel_size)
        self.rebuilt = conv(n_feats, args.n_colors, kernel_size)

    def forward(self, x):
        out = self.trans(x)
        out = self.head(out)
        out = self.rdb(out)
        out = self.tail(out)
        out = self.rebuilt(out)
        out = x - out
        return out
        
class MLEGN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MLEGN, self).__init__()
        n_feats = args.n_feats
        kernel_size = args.kernel_size
        block = args.block
        n_layer = args.n_layer
        
        self.noise_head = conv(args.n_colors, n_feats, kernel_size) 
        self.edge_head = conv(args.n_colors, n_feats, kernel_size) 
        self.Edge_Net = Edge_Net(args, n_feats, n_layer)

        self.image_feature = FEB(args, n_layer)
        self.edge_feature = FEB(args, n_layer)
        self.image_rg_1 = RG(n_feats, block)
        self.image_rg_2 = RG(n_feats, block)
        self.edge_rg_1 = RG(n_feats, block)
        self.edge_rg_2 = RG(n_feats, block)
        self.cat_rg_1 = RG(n_feats, block)
        self.cat_rg_2 = RG(n_feats, block)
        self.cat_rg_3 = RG(n_feats, block)
        self.fusion_1 = nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)
        self.fusion_2 = nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)
        self.fusion_3 = nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)
        
        self.tail = conv(n_feats, args.n_colors, kernel_size)


    def forward(self, x):
        noise_map = self.noise_head(x)

        edge = self.Edge_Net(x)
        edge_map = self.edge_head(edge)
        
        ##特征编码
        image_feature_1 = self.image_feature(noise_map)
        edge_feature_1 = self.edge_feature(edge_map)
        ##第一次edge attention
        leve_l = image_feature_1 + edge_feature_1
        
        image_feature_2 = self.image_rg_1(image_feature_1)
        edge_feature_2 = self.edge_rg_1(edge_feature_1)
        ##第二次edge guided
        leve_2 = image_feature_2 + edge_feature_2
        
        image_feature_3 = self.image_rg_2(edge_feature_2)
        edge_feature_3 = self.edge_rg_2(edge_feature_2)
        ##第三次edge guided
        leve_3 = image_feature_3 + edge_feature_3
        
        cat_1 = torch.cat([leve_l, leve_2], 1)
        cat_1 = self.fusion_1(cat_1)
        cat_1 = self.cat_rg_1(cat_1)
        
        cat_2 = torch.cat([leve_2, leve_3], 1)
        cat_2 = self.fusion_2(cat_2)
        cat_2 = self.cat_rg_2(cat_2)
        
        cat_3 = torch.cat([cat_1, cat_2], 1)
        cat_3 = self.fusion_3(cat_3)
        out = self.cat_rg_3(cat_3)
        
        out = self.tail(out)
        return edge, out
