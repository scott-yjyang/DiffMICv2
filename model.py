import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50, resnet101
from torchvision.models.densenet import densenet169

from timm.models import create_model
from pretraining.dcg import DCG
import numpy as np
from einops import rearrange, reduce

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out

class ConditionalConv2d(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalConv2d, self).__init__()
        self.num_out = num_out
        self.lin = nn.Conv2d(num_in, num_out, kernel_size=1, stride=1)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        bz, dim, h, w = out.size()
        gamma = self.embed(t)
        gamma = gamma.reshape(bz, h*w, dim).permute(0,2,1).reshape(bz,dim,h,w)
        out = gamma*out
        return out


from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vits, build_efficient_sam_vitt

class DenoiseUNet(nn.Module):
    def __init__(self, y_dim, feature_dim, n_steps, guidance=True):
        super(DenoiseUNet, self).__init__()

        if guidance:
            self.lin1 = ConditionalConv2d(y_dim * 2, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalConv2d(y_dim, feature_dim, n_steps)

        self.unetnorm1 = nn.BatchNorm2d(feature_dim)
        self.lin2 = ConditionalConv2d(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm2d(feature_dim)
        self.lin3 = ConditionalConv2d(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm2d(feature_dim)
        self.lin4 = nn.Conv2d(feature_dim, y_dim, kernel_size=1, stride=1)
        self.cond_weight = nn.Parameter(torch.randn((1, feature_dim, 7)), requires_grad=True)

        
    def forward(self, x, y, t):

        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        skip_y1 = y

        w = torch.softmax(self.cond_weight,dim=2)
        x_weight = torch.sum(x*w,dim=-1)
        # print(x_weight.shape)
        y = x_weight.unsqueeze(-1).unsqueeze(-1) * y
        # y = x*y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        
        # y = x * y
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        y = y + skip_y1

        y = self.lin4(y)
            
        return y


class ConditionalModel(nn.Module):
    def __init__(self, config, guidance=True):
        super(ConditionalModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        data_dim = config.model.data_dim
        y_dim = config.data.num_classes
        arch = config.model.arch
        feature_dim = config.model.feature_dim
        hidden_dim = config.model.hidden_dim
        self.guidance = guidance
        
        # encoder for x
        self.encoder_x = SamEncoder(arch=arch, feature_dim=feature_dim, config=config)
        self.norm = nn.BatchNorm1d(feature_dim)

        self.encoder_x_l = ResNetEncoder(arch=arch, feature_dim=feature_dim, config=config, local=True)
        self.norm_l = nn.BatchNorm1d(feature_dim)

        # gated attention
        # self.gate = nn.Sequential(
        #     nn.Linear(feature_dim * 2, feature_dim),
        #     nn.Sigmoid()
        # )
        
        if self.guidance:
            self.lin1 = ConditionalConv2d(y_dim * 2, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalConv2d(y_dim, feature_dim, n_steps)
        self.unetnorm1 = nn.BatchNorm2d(feature_dim)
        self.lin2 = ConditionalConv2d(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm2d(feature_dim)
        self.lin3 = ConditionalConv2d(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm2d(feature_dim)
        self.lin4 = nn.Conv2d(feature_dim, y_dim, kernel_size=1, stride=1)
        self.cond_weight = nn.Parameter(torch.randn((1, feature_dim, 7)), requires_grad=True)

        
    def forward(self, x, y, t, x_l, attn):
        bz, np, I, J = x_l.shape

        x_l = x_l.view(bz * np, I, J).unsqueeze(1).expand(-1, 3, -1 , -1)
        x_l = self.encoder_x_l(x_l)
        x_l = self.norm_l(x_l)
        
        x = self.encoder_x(x)
        x = self.norm(x)


        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        
        x_l = x_l.reshape(bz , np, x_l.shape[1]).permute(0,2,1)
        x = torch.cat([x.unsqueeze(-1),x_l],dim=-1)
        w = torch.softmax(self.cond_weight,dim=2)
        # print(w.shape)
        # print(x.shape)
        x_weight = torch.sum(x*w,dim=-1)
        # print(x_weight.shape)
        y = x_weight.unsqueeze(-1).unsqueeze(-1) * y
        
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        y = self.lin4(y)
            
        return y




# ResNet 18 or 50 as image encoder
class ResNetEncoder(nn.Module):
    def __init__(self, arch='resnet18', feature_dim=128, config=None, local=False):
        super(ResNetEncoder, self).__init__()

        self.f = []
        #print(arch)
        if arch == 'resnet50':
            backbone = resnet50()
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'resnet18':
            backbone = resnet18()
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'densenet121':
            backbone = densenet121(pretrained=True)
            self.featdim = backbone.classifier.weight.shape[1]
        elif arch == 'vit':
            backbone = create_model('pvt_v2_b2',
                pretrained=True,
                num_classes=4,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            backbone.head = nn.Sequential()
            self.featdim = 512

            

        for name, module in backbone.named_children():
            if name != 'fc':
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        
        self.g = nn.Linear(self.featdim, feature_dim)


    def forward_feature(self, x):
        feature = self.f(x)
        feature = torch.flatten(feature, start_dim=1)
        feature = self.g(feature)

        return feature

    def forward(self, x):
        feature = self.forward_feature(x)
        return feature

class SamEncoder(nn.Module):
    def __init__(self, arch='resnet18', feature_dim=128, config=None, image_size=224):
        super(SamEncoder, self).__init__()

        self.f = []
        self.f = build_efficient_sam_vits(image_size=image_size)
        self.featdim = 384
        
            
        self.g = nn.Conv2d(self.featdim, feature_dim, kernel_size=1, stride=1)

    def forward_feature(self, x):
        feature = self.f(x)
        feature = self.g(feature)
        feature = F.adaptive_avg_pool2d(feature,(1,1))
        return torch.flatten(feature,start_dim=1)

    def forward(self, x):
        feature = self.forward_feature(x)
        return feature

