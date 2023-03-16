
import torch
from torch import nn as nn
import torch.nn.functional as F







class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x




class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Conv2d(dim,dim,1)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=5//2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=((5//2)*3), groups=dim, dilation=3)
        self.conv_v = nn.Conv2d(dim,dim,(1,5),1,[2,0],groups=dim)
        self.conv_h = nn.Conv2d(dim, dim,(5,1),1,[0,2],groups=dim)
        self.conv_v_2 = nn.Conv2d(dim,dim,(1,3),1,[1,0],groups=dim)
        self.conv_h_2 = nn.Conv2d(dim, dim,(3,1),1,[0,1],groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        x = self.head(x)
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        x_hv = self.conv_h(self.conv_v(x))
        x_hv2 = self.conv_h_2(self.conv_v_2(x))
        attn = self.conv1(attn+x_hv+x+x_hv2)

        return u * attn




class Attention(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.norm = LayerNorm(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.proj_1 = nn.Conv2d(n_feats, n_feats, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(n_feats)
        self.proj_2 = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(self.norm(x))
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x*self.scale + shorcut
        return x




class MLP(nn.Module):
    def __init__(self, n_feats):
        super().__init__()

        self.norm = LayerNorm(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        i_feats = 2*n_feats

        self.fc1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.activation = nn.GELU()
        self.dwconv = nn.Conv2d(i_feats, i_feats, 3, 1, 1, bias=False, groups=i_feats)
        self.fc2 = nn.Conv2d(i_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)

        x = self.fc1(x)
        x0=x
        x= self.dwconv(x)
        x = x+x0
        x=self.activation(x)

        x = self.fc2(x)

        return x*self.scale + shortcut



class BasicConv(nn.Module):
    def __init__(self, c):
        super(BasicConv, self).__init__()
        self.atten = Attention(c)
        self.MLP = MLP(c)

    def forward(self,x):
        x= self.MLP(self.atten(x))
        return x



class LGVAN(nn.Module):

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=10,
                 upscale=4,):
        super(LGVAN, self).__init__()


        self.scale = upscale
        self.num_block = num_block

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)



        self.layers = nn.ModuleList()

        for i_layer in range(self.num_block):
            layer = BasicConv(num_feat)
            self.layers.append(layer)





        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1,groups=2)

        if self.scale == 4:
            self.upsapling = nn.Sequential(
                nn.Conv2d(num_feat, num_feat*4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(num_feat, num_feat*4, 1, 1, 0),
                nn.PixelShuffle(2)
            )
        else:
            self.upsapling = nn.Sequential(
                nn.Conv2d(num_feat, num_feat*self.scale*self.scale, 1, 1, 0),
                nn.PixelShuffle(self.scale)
            )

        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)


    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    def forward(self, x):
        x0=x

        x = self.conv_first(x)

        x1=self.forward_features(x)

        res = self.conv_after_body(x1)
        res += x

        x = self.conv_last(self.act(self.upsapling(res)))
        x_i = F.interpolate(x0, scale_factor=self.scale, mode='bilinear', align_corners=False)


        return x+x_i

