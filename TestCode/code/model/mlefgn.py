from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return MLEFGN(args)

## define the basic component of RDB
class DB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kernel_size = 3):
        super(DB_Conv, self).__init__()
        n_feats = inChannels
        rate  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(n_feats, rate, kernel_size, padding=(kernel_size-1)//2, stride=1),
            nn.ReLU()
        ])
    
    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


## define the dense block (DB)
class DB(nn.Module):
    def __init__(self, args, n_layer):
        super(DB, self).__init__()
        n_feats = args.n_feats
        rate  = args.rate
        kernel_size = args.kernel_size
        
        convs = []
        for n in range(n_layer):
            convs.append(DB_Conv(n_feats + n * rate, rate))
        self.convs = nn.Sequential(*convs)
        
        self.LFF = nn.Conv2d(n_feats + n_layer * rate, n_feats, 1, padding=0, stride=1)
    
    def forward(self, x):
        out = self.LFF(self.convs(x)) + x
        return out

## define the residual group (RG)
class RG(nn.Module):
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

## define the Edge-Net
class Edge_Net(nn.Module):
    def __init__(self, args, n_feats, n_layer, kernel_size=3, conv=common.default_conv):
        super(Edge_Net, self).__init__()

        self.trans = conv(args.n_colors, n_feats, kernel_size)
        self.head = common.ResBlock(conv, n_feats, kernel_size)
        self.rdb = DB(args, n_layer)
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
        
class MLEFGN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MLEFGN, self).__init__()
        n_feats = args.n_feats
        kernel_size = args.kernel_size
        block = args.block
        n_layer = args.n_layer
        

        ## Stage I
        self.noise_head = conv(args.n_colors, n_feats, kernel_size) 
        self.edge_head = conv(args.n_colors, n_feats, kernel_size) 
        self.Edge_Net = Edge_Net(args, n_feats, n_layer)


        ## Stage II
        self.image_feature = DB(args, n_layer)
        self.edge_feature = DB(args, n_layer)

        self.image_rg_1 = RG(n_feats, block)
        self.edge_rg_1 = RG(n_feats, block)

        self.image_rg_2 = RG(n_feats, block)
        self.edge_rg_2 = RG(n_feats, block)


        ## Stage III
        self.fusion_1 = nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)
        self.cat_rg_1 = RG(n_feats, block)
        self.fusion_2 = nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)
        self.cat_rg_2 = RG(n_feats, block)
        self.fusion_3 = nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)
        self.cat_rg_3 = RG(n_feats, block)
        self.tail = conv(n_feats, args.n_colors, kernel_size)


    def forward(self, x):

        ## Stage I: Edge Reconstruction
        noise_map = self.noise_head(x)

        edge = self.Edge_Net(x)
        edge_map = self.edge_head(edge)
        
        ## Stage II: Feature Extraction
        image_feature_1 = self.image_feature(noise_map)
        edge_feature_1 = self.edge_feature(edge_map)

        image_feature_2 = self.image_rg_1(image_feature_1)
        edge_feature_2 = self.edge_rg_1(edge_feature_1)

        image_feature_3 = self.image_rg_2(image_feature_2)
        edge_feature_3 = self.edge_rg_2(edge_feature_2)


        ## Stage III: Edge Guidance Image Reconstruction
        leve_l = image_feature_1 + edge_feature_1
        leve_2 = image_feature_2 + edge_feature_2
        leve_3 = image_feature_3 + edge_feature_3
        
        cat_1 = torch.cat([leve_l, leve_2], 1)
        cat_1 = self.fusion_1(cat_1)
        cat_1 = self.cat_rg_1(cat_1)
        
        cat_2 = torch.cat([leve_2, leve_3], 1)
        cat_2 = self.fusion_2(cat_2)
        cat_2 = self.cat_rg_2(cat_2)
        
        cat_3 = torch.cat([cat_1, cat_2], 1)
        cat_3 = self.fusion_3(cat_3)
        denoised = self.cat_rg_3(cat_3)
        
        denoised = self.tail(denoised)
        return denoised, edge