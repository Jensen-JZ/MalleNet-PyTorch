import torch
import numpy as np
import torch.nn as nn
# import bilateral_slice
import torch.nn.functional as F
from collections import OrderedDict


# like "tf.nn.space_to_depth"
def space_2_depth(x, down_scale):
    '''Space_to_depth for PyTorch.
        Args: x, input tensor, 4D, [batch, channel, height, width]
              down_scale, down scale factor
        Return: out tensor, 4D, [batch, channel*down_scale*down_scale, height/down_scale, width/down_scale]
    '''
    B, C, H, W = x.shape
    out_channel = C * (down_scale**2)
    height = H // down_scale
    width = W // down_scale
    
    x = x.view(B*C, height, down_scale, width, down_scale)
    out = x.permute(0, 2, 4, 1, 3).contiguous().view(B, out_channel, height, width)
    
    return out


class ShufflePyramidDecom(nn.Module):
    '''Shuffle Pyramid Decomposition
        Args: x, input tensor, 4D, [batch, channel, height, width]
        Return: pyramids, list of tensors, 4D, [batch, channel*down_scale*down_scale, height/down_scale, width/down_scale]
    '''
    def __init__(self, down_scale, depth=3):
        super().__init__()
        self.down_scale = down_scale
        self.depth = depth
        
    def __call__(self, x):
        current = x
        pyramids = []
        pyramids.append(current)
        for i in range(self.depth):
            current = space_2_depth(current, self.down_scale)
            pyramids.append(current)
        
        return pyramids


# For more details, check the paper "HDRNet"
# class BilateralSliceApply(nn.Module):
#     '''Bilateral Slice Apply
#         Args: bilateral_grid, tensor, 5D, [batch, channel, height, width, depth]
#               guide, tensor, 3D, [batch, height, width]
#               x, tensor, 4D, [batch, channel, height, width]
#         Returns: out, tensor, 4D, [batch, channel, height, width]
#     '''
#     def __init__(self):
#         super().__init__()
#         self.bilateral_slice_apply = bilateral_slice
        
#     def forward(self, bilateral_grid, guide, x, has_offset=False):
#         return self.bilateral_slice_apply.forward(bilateral_grid, guide, x, has_offset)
## >>> PyTorch version of "Bilateral Slice Apply"
class BilateralSlice(nn.Module):
    def __init__(self):
        super(BilateralSlice, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        device = bilateral_grid.get_device()
        guidemap = guidemap.unsqueeze(1)

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)

        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):
        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''
        # R = torch.sum(full_res_input * coeff[:, 0:1, :, :], dim=1, keepdim=True) + coeff[:, 1:2, :, :]
        # G = torch.sum(full_res_input * coeff[:, 0:1, :, :], dim=1, keepdim=True) + coeff[:, 1:2, :, :]
        # B = torch.sum(full_res_input * coeff[:, 0:1, :, :], dim=1, keepdim=True) + coeff[:, 1:2, :, :]
        # return torch.cat([R, G, B], dim=1)
        out = torch.sum(full_res_input * coeff[:, 0:1, :, :], dim=1, keepdim=True) + coeff[:, 1:2, :, :]
        
        return out


class BilateralSliceApply(nn.Module):
    def __init__(self):
        super().__init__()
        self.bilateral_slice = BilateralSlice()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, bilateral_grid, guidemap, full_res_input, has_offset=True):
        coeff = self.bilateral_slice(bilateral_grid.permute(0, 1, 4, 2, 3), guidemap)
        out = self.apply_coeffs(coeff, full_res_input)

        return out
## PyTorch version of "Bilateral Slice Apply" <<< ##


class ResConvBlock(nn.Module):
    '''Inverse Residual Bottleneck Block
        Args: x, tensor, 4D, [batch, channel, height, width]
        Return: out, tensor, 4D, [batch, channel, height, width]
    '''
    def __init__(self, in_channel=64, norm='kaiming'):
        super().__init__()
        self.conv_block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channel, in_channel*5, kernel_size=1, padding=0, bias=True)), 
            ('norm1', nn.BatchNorm2d(in_channel*5)),
            ('prelu', nn.PReLU(init=0.0)), 
            # Depth-wise Conv
            ('dwconv', nn.Conv2d(in_channel*5, in_channel*5, kernel_size=3, padding=1, bias=True, groups=in_channel*5)),
            ('norm2', nn.BatchNorm2d(in_channel*5)), 
            ('prelu', nn.PReLU(init=0.0)),
            ('conv2', nn.Conv2d(in_channel*5, in_channel, kernel_size=1, bias=True)), 
            ('norm3', nn.BatchNorm2d(in_channel)),
            ('prelu', nn.PReLU(init=0.0)),
        ]))
        self.init_weights(norm)
        
    def init_weights(self, norm='kaiming'):
        for _, param in self.named_modules():
            if isinstance(param, nn.Conv2d):
                if norm == 'kaiming':
                    nn.init.kaiming_normal_(param.weight)
        
        
    def forward(self, x):
        identity = x
        out = identity + self.conv_block(x)
        
        return out


class Policy(nn.Module):
    '''Policy Network: Efficient Perdictor Net for MalleNet
        Args: x, list of tensors
        Return: out, tensor, 6D, [batch, channel, height, width, depth, group]
    '''
    def __init__(self, in_channel=64, num_feature=64, n_in=1, n_out=2, gz=1, stage=3, norm='kaiming'):
        super().__init__()
        self.stage = stage
        self.n_in = n_in
        self.n_out = n_out
        self.in_channel = in_channel
        self.num_feature = num_feature
        self.gz = gz
        
        self.low_dense_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel, num_feature, 3, padding=1, bias=True)),
            ('prelu', nn.PReLU(init=0.0)),
        ]))
        res_block_1 = [
            ResConvBlock(in_channel=num_feature)
            for i in range(3)
        ]
        self.res_block_1 = nn.Sequential(*res_block_1)
        
        if self.stage > 1:
            self.low_pool_1 = nn.MaxPool2d(kernel_size=2)
            res_block_2 = [
                ResConvBlock(in_channel=num_feature)
                for i in range(3)
            ]
            self.res_block_2 = nn.Sequential(*res_block_2)
            
        if self.stage > 2:
            self.low_pool_2 = nn.MaxPool2d(kernel_size=2)
            res_block_3 = [
                ResConvBlock(in_channel=num_feature)
                for i in range(3)
            ]
            self.res_block_3 = nn.Sequential(*res_block_3)
            
        if self.stage > 3:
            self.low_pool_3 = nn.MaxPool2d(kernel_size=2)
            res_block_4 = [
                ResConvBlock(in_channel=num_feature)
                for i in range(3)
            ]
            self.res_block_4 = nn.Sequential(*res_block_4)
            
        self.low_dense_2 = nn.Conv2d(num_feature, self.gz*self.n_in*self.n_out, 1, padding=0, bias=True)
        self.init_weights(norm)
            
    def init_weights(self, norm='kaiming'):
        for name, param in self.named_modules():
            if isinstance(param, nn.Conv2d) and name.find('res')==-1:
                if norm == 'kaiming':
                    nn.init.kaiming_normal_(param.weight)
            
    def forward(self, x):
        x = self.low_dense_1(x)
        x = self.res_block_1(x)
        if self.stage > 1:
            x = self.low_pool_1(x)
            x = self.res_block_2(x)
        if self.stage > 2:
            x = self.low_pool_2(x)
            x = self.res_block_3(x)
        if self.stage > 3:
            x = self.low_pool_3(x)
            x = self.res_block_4(x)
        out = self.low_dense_2(x)
        out = torch.stack(torch.split(
            out, out.shape[1]//(self.n_in*self.n_out), dim=1
        ), dim=2).permute(0, 2, 3, 4, 1)
        
        return out
    
    
class LargePolicy(nn.Module):
    '''Larger Policy Network: Efficient Perdictor Net for MalleNet
        Args: x, list of tensors
        Return: out, tensor, 6D, [batch, channel, height, width, depth, group]
    '''
    def __init__(self, in_channel=64, num_feature=64, n_in=1, n_out=2, gz=1, stage=3, norm='kaiming'):
        super().__init__()
        self.stage = stage
        self.n_in = n_in
        self.n_out = n_out
        self.in_channel = in_channel
        self.num_feature = num_feature
        self.gz = gz
        
        self.low_dense_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel, num_feature, 3, padding=1, bias=True)),
            ('prelu', nn.PReLU(init=0.0)),
        ]))
        res_block_1 = [
            ResConvBlock(in_channel=num_feature)
            for i in range(3)
        ]
        self.res_block_1 = nn.Sequential(*res_block_1)
        
        if self.stage > 1:
            self.low_pool_1 = nn.Sequential(OrderedDict([
                ('pool', nn.MaxPool2d(kernel_size=2)),
                ('conv', nn.Conv2d(num_feature, num_feature*2, 3, padding=1, bias=True)),
                ('norm', nn.BatchNorm2d(num_feature*2)),
                ('prelu', nn.PReLU(init=0.0)),
            ]))
            res_block_2 = [
                ResConvBlock(num_feature*2)
                for i in range(3)
            ]
            self.res_block_2 = nn.Sequential(*res_block_2)
            
        if self.stage > 2:
            self.low_pool_2 = nn.Sequential(OrderedDict([
                ('pool', nn.MaxPool2d(kernel_size=2)),
                ('conv', nn.Conv2d(num_feature*2, num_feature*4, 3, padding=1, bias=True)),
                ('norm', nn.BatchNorm2d(num_feature*4)),
                ('prelu', nn.PReLU(init=0.0)),
            ]))
            res_block_3 = [
                ResConvBlock(num_feature*4)
                for i in range(3)
            ]
            self.res_block_3 = nn.Sequential(*res_block_3)
            
        if self.stage > 3:
            self.low_pool_3 = nn.Sequential(OrderedDict([
                ('pool', nn.MaxPool2d(kernel_size=2)),
                ('conv', nn.Conv2d(num_feature*4, num_feature*8, 3, padding=1, bias=True)),
                ('norm', nn.BatchNorm2d(num_feature*8)),
                ('prelu', nn.PReLU(init=0)),
            ]))
            res_block_4 = [
                ResConvBlock(num_feature*8)
                for i in range(3)
            ]
            self.res_block_4 = nn.Sequential(*res_block_4)
            
        self.stage = 4 if self.stage > 3 else self.stage
        self.low_dense_2 = nn.Conv2d(num_feature*2**(self.stage-1), self.n_in*self.n_out*self.gz, 1, padding=0, bias=True)
        self.init_weights(norm)
        
    def init_weights(self, norm='kaiming'):
        for name, param in self.named_modules():
            if isinstance(param, nn.Conv2d) and name.find('res')==-1:
                if norm == 'kaiming':
                    nn.init.kaiming_normal_(param.weight)
        
    def forward(self, x):
        x = self.low_dense_1(x)
        x = self.res_block_1(x)
        if self.stage > 1:
            x = self.low_pool_1(x)
            x = self.res_block_2(x)
        if self.stage > 2:
            x = self.low_pool_2(x)
            x = self.res_block_3(x)
        if self.stage > 3:
            x = self.low_pool_3(x)
            x = self.res_block_4(x)
        out = self.low_dense_2(x)
        out = torch.stack(torch.split(
            out, out.shape[1]//(self.n_in*self.n_out), dim=1
        ), dim=2).permute(0, 2, 3, 4, 1)
        
        return out


class ModelOne(nn.Module):
    '''Model One: Pyramid Decom of MalleNet
        Args: x, input tensor, 4D, [batch, channel, height, width]
        Return: out, list of tensors
    '''
    def __init__(self, in_channel=3, num_feature=64, down_scale=2, depth=3, norm='kaiming'):
        super().__init__()
        self.in_channel = in_channel
        self.num_feature = num_feature
        self.down_scale = down_scale
        self.depth = depth
        
        factors = [(self.down_scale**2)**i for i in range(self.depth+1)]
        
        self.shuffle_pyramid_decom = ShufflePyramidDecom(down_scale=self.down_scale, depth=self.depth)
        self.pyramids = []
        self.conv_pyramid_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel*factors[0], num_feature, 3, padding=1, bias=True)),
            ('res', ResConvBlock(num_feature)),   
        ]))
        self.conv_pyramid_2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel*factors[1], num_feature, 3, padding=1, bias=True)),
            ('res', ResConvBlock(num_feature)),
        ]))
        self.conv_pyramid_3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel*factors[-2], num_feature*2, 3, padding=1, bias=True)),
            ('res', ResConvBlock(num_feature*2)),
        ]))
        self.conv_pyramid_4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel*factors[-1], num_feature*4, 3, padding=1, bias=True)),
            ('res', ResConvBlock(num_feature*4)),
        ]))
        self.pyramids += [
            self.conv_pyramid_1, self.conv_pyramid_2, self.conv_pyramid_3, self.conv_pyramid_4
        ]
        self.init_weights(norm)
        
    def init_weights(self, norm='kaiming'):
        for name, param in self.named_modules():
            if isinstance(param, nn.Conv2d) and name.find('res')==-1:
                if norm == 'kaiming':
                    nn.init.kaiming_normal_(param.weight)

    def forward(self, x):
        pyramid_outs = self.shuffle_pyramid_decom(x)
        outs = []
        for i, feature in enumerate(pyramid_outs):
            outs.append(self.pyramids[i](feature))

        return outs, pyramid_outs


class ModelTwo(nn.Module):
    '''Model Two: Backbone of MalleNet
        Args: x, list of tensors
        Return: out, list of tensors
    ''' 
    def __init__(self, num_feature=64, low_res='down', stage=3, n_in=1, n_out=2, gz=1, group=None, norm='kaiming'):
        super().__init__()
        self.num_feature = num_feature
        self.low_res = low_res
        self.stage = stage
        self.n_in = n_in
        self.n_out = n_out
        self.gz = gz
        self.group = num_feature if group is None else group
        self.bilateral_slice_apply = BilateralSliceApply()
        
        self.factors = [1]
        self.factors += [2**i for i in range(stage)]
        
        if self.low_res == 'down':
            low_res_blocks = [
                Policy(in_channel=self.factors[i]*num_feature, num_feature=num_feature, 
                       n_in=self.n_in, n_out=self.n_out, gz=self.factors[i]*self.group*self.gz, 
                       stage=self.stage) for i in range(self.stage + 1)
            ]
            self.low_res_blocks = nn.Sequential(*low_res_blocks)
                        
        elif self.low_res == 'downavg2':
            low_res_blocks = [
                nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(kernel_size=2, stride=2)), 
                ('policy', Policy(in_channel=self.factors[i]*num_feature, 
                                  num_feature=num_feature, n_in=self.n_in, 
                                  n_out=self.n_out, gz=self.factors[i]*self.group*self.gz, 
                                  stage=self.stage)),
                ])) for i in range(2)
            ]
            low_res_blocks += [
                Policy(in_channel=self.factors[i]*num_feature, num_feature=num_feature, 
                       n_in=self.n_in, n_out=self.n_out, gz=self.factors[i]*self.group*self.gz, 
                       stage=self.stage) for i in range(2, self.stage+1)
            ]
            self.low_res_blocks = nn.Sequential(*low_res_blocks)
            
        elif self.low_res == 'ldownavg2':
            low_res_blocks = [
                nn.Sequential(OrderedDict([
                    ('avg', nn.AvgPool2d(kernel_size=2, stride=2)), 
                    ('largepolicy', LargePolicy(in_channel=self.factors[i]*num_feature, 
                                                num_feature=num_feature, n_in=self.n_in, 
                                                n_out=self.n_out, gz=self.factors[i]*self.group*self.gz, 
                                                stage=self.stage)), 
                ])) for i in range(2)
            ]
            low_res_blocks += [
                LargePolicy(in_channel=self.factors[i]*num_feature, num_feature=num_feature, 
                            n_in=self.n_in, n_out=self.n_out, gz=self.factors[i]*self.group*self.gz, 
                            stage=self.stage) for i in range(2, self.stage+1)
            ]
            self.low_res_blocks = nn.Sequential(*low_res_blocks)
            
        elif self.low_res == 'downavg4':
            low_res_blocks = [
                nn.Sequential(OrderedDict([
                    ('avg', nn.AvgPool2d(kernel_size=4, stride=4)), 
                    ('policy', Policy(in_channel=self.factors[i]*num_feature, 
                                      num_feature=num_feature, n_in=self.n_in, 
                                      n_out=self.n_out, gz=self.factors[i]*self.group*self.gz, 
                                      stage=self.stage)),
                ])) for i in range(2)
            ]
            low_res_blocks += [
                Policy(in_channel=self.factors[i]*num_feature, num_feature=num_feature, 
                       n_in=self.n_in, n_out=self.n_out, gz=self.factors[i]*self.group*self.gz, 
                       stage=self.stage) for i in range(2, self.stage+1)
            ]
            self.low_res_blocks = nn.Sequential(*low_res_blocks)
            
        elif self.low_res == 'ldownavg4':
            low_res_blocks = [
                nn.Sequential(OrderedDict([
                    ('avg', nn.AvgPool2d(kernel_size=4, stride=4)),
                    ('largepolicy', LargePolicy(in_channel=self.factors[i]*num_feature, 
                                                num_feature=num_feature, n_in=self.n_in, 
                                                n_out=self.n_out, gz=self.factors[i]*self.group*self.gz, 
                                                stage=self.stage)),
                ])) for i in range(2)
            ]
            low_res_blocks += [
                LargePolicy(in_channel=self.factors[i]*num_feature, num_feature=num_feature, 
                            n_in=self.n_in, n_out=self.n_out, gz = self.factors[i]*self.group*self.gz, 
                            stage=self.stage) for i in range(2, self.stage+1)
            ]
            self.low_res_blocks = nn.Sequential(*low_res_blocks)
            
        bilateral_blocks = [
            nn.Sequential(OrderedDict([
                ('norm', nn.BatchNorm2d(self.factors[i]*num_feature)),
                ('prelu', nn.PReLU(init=0.0)),
            ])) for i in range(self.stage+1)
        ]
        self.bilateral_blocks = nn.Sequential(*bilateral_blocks)
        self.gate = [torch.tensor([0.0], requires_grad=True) for i in range(4)]
        
        self.conv_4_1 = nn.Sequential(OrderedDict([
            ('res1', ResConvBlock(num_feature*self.factors[-1])),
            ('res2', ResConvBlock(num_feature*self.factors[-1])),
        ]))
        self.conv_4_2 = nn.Sequential(OrderedDict([
            ('res1', ResConvBlock(num_feature*self.factors[-1])),
            ('res2', ResConvBlock(num_feature*self.factors[-1])),
        ]))
        self.conv_3_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(num_feature*(self.factors[-1] + self.factors[-2]), num_feature*self.factors[-2], 
                               3, padding=1, bias=True)),
            ('nrom', nn.BatchNorm2d(num_feature*self.factors[-2])),
            ('res1', ResConvBlock(num_feature*self.factors[-2])),
            ('res2', ResConvBlock(num_feature*self.factors[-2])),
        ]))
        self.conv_3_2 = nn.Sequential(OrderedDict([
            ('res1', ResConvBlock(num_feature*self.factors[-2])),
            ('res2', ResConvBlock(num_feature*self.factors[-2])),
        ]))
        self.conv_2_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(num_feature*(self.factors[-2] + self.factors[1]), num_feature, 3, padding=1, bias=True)),
            ('norm', nn.BatchNorm2d(num_feature)),
            ('res1', ResConvBlock(num_feature)),
            ('res2', ResConvBlock(num_feature)),
        ]))
        self.conv_2_2 = nn.Sequential(OrderedDict([
            ('res1', ResConvBlock(num_feature)),
            ('res2', ResConvBlock(num_feature)),
        ]))
        self.conv_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(num_feature*(self.factors[0] + self.factors[1]), num_feature, 1, padding=0, bias=True)),
            ('norm', nn.BatchNorm2d(num_feature)),
        ]))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.init_weights(norm)
        
    def init_weights(self, norm='kaiming'):
        for name, param in self.named_modules():
            if isinstance(param, nn.Conv2d) and name.find('res')==-1:
                nn.init.kaiming_normal_(param.weight)
        
    def forward(self, x, has_offset=True):
        inputs, pyramids = x
        outs = []
        image_1, image_2, image_3, image_4 = inputs
        # level_4
        image_4 = self.conv_4_1(image_4)
        # **** MalleConv ****
        group = self.group * self.factors[-1]
        content_feature = image_4
        # *** Efficient Predictor Net ***
        grid_coefficients = self.low_res_blocks[-1](content_feature)
        grid_coefficients = torch.stack(torch.split(
            grid_coefficients, grid_coefficients.shape[4]//group, dim=4
        ), dim=5)
        # *** Efficient Predictor Net ***
        post_image = []
        # ** Efficient On-the-fly slicing **
        for j in range(group):
            post_image_j = self.bilateral_slice_apply(
                grid_coefficients[:, :, :, :, :, j].contiguous(),
                torch.clip(content_feature[:, j, :, :], 0, 1), 
                content_feature[:, j:j+1, :, :].contiguous(),
                has_offset
            ) 
            post_image.append(post_image_j)
        post_image = torch.cat(post_image, dim=1)
        post_image = self.bilateral_blocks[-1](post_image)
        post_image = torch.clip(post_image, -1, 1)
        image_4 = image_4 + post_image * self.gate[-1].to(post_image.device)
        # ** Efficient On-the-fly slicing **
        image_4 = self.conv_4_2(image_4)
        # **** MalleConv ****        
        
        # level_3
        image_3 = self.conv_3_1(torch.cat([image_3, self.upsample(image_4)], dim=1))
        group = self.group * self.factors[-2]
        content_feature = image_3
        grid_coefficients = self.low_res_blocks[-2](content_feature)
        grid_coefficients = torch.stack(torch.split(
            grid_coefficients, grid_coefficients.shape[4]//group, dim=4
        ), dim=5)
        post_image = []
        for j in range(group):
            post_image_j = self.bilateral_slice_apply(
                grid_coefficients[:, :, :, :, :, j].contiguous(),
                torch.clip(content_feature[:, j, :, :], 0, 1), 
                content_feature[:, j:j+1, :, :].contiguous(),
                has_offset
            ) 
            post_image.append(post_image_j)
        post_image = torch.cat(post_image, dim=1)
        post_image = self.bilateral_blocks[-2](post_image)
        post_image = torch.clip(post_image, -1, 1)
        image_3 = image_3 + post_image * self.gate[-2].to(post_image.device)
        image_3 = self.conv_3_2(image_3)
        
        # level_2
        image_2 = self.conv_2_1(torch.cat([image_2, self.upsample(image_3)], dim=1))
        group = self.group * self.factors[1]
        content_feature = image_2
        grid_coefficients = self.low_res_blocks[1](content_feature)
        grid_coefficients = torch.stack(torch.split(
            grid_coefficients, grid_coefficients.shape[4]//group, dim=4
        ), dim=5)
        post_image = []
        for j in range(group):
            post_image_j = self.bilateral_slice_apply(
                grid_coefficients[:, :, :, :, :, j].contiguous(),
                torch.clip(content_feature[:, j, :, :], 0, 1), 
                content_feature[:, j:j+1, :, :].contiguous(),
                has_offset
            ) 
            post_image.append(post_image_j)
        post_image = torch.cat(post_image, dim=1)
        post_image = self.bilateral_blocks[1](post_image)
        post_image = torch.clip(post_image, -1, 1)
        image_2 = image_2 + post_image * self.gate[1].to(post_image.device)
        image_2 = self.conv_2_2(image_2)
        
        # level_1
        image_1 = self.conv_1(torch.cat([image_1, self.upsample(image_2)], dim=1))
        group = self.group * self.factors[0]
        content_feature = image_1
        grid_coefficients = self.low_res_blocks[0](content_feature)
        grid_coefficients = torch.stack(torch.split(
            grid_coefficients, grid_coefficients.shape[4]//group, dim=4
        ), dim=5)
        post_image = []
        for j in range(group):
            post_image_j = self.bilateral_slice_apply(
                grid_coefficients[:, :, :, :, :, j].contiguous(),
                torch.clip(content_feature[:, j, :, :], 0, 1), 
                content_feature[:, j:j+1, :, :].contiguous(),
                has_offset
            ) 
            post_image.append(post_image_j)
        post_image = torch.cat(post_image, dim=1)
        post_image = self.bilateral_blocks[0](post_image)
        post_image = torch.clip(post_image, -1, 1)
        image_1 = image_1 + post_image * self.gate[0].to(post_image.device)
        
        outs.append(image_1)
        outs.append(image_2)        
        outs.append(image_3)        
        outs.append(image_4)
        
        return outs, pyramids


class ModelThree(nn.Module):
    '''Model Three: End of MalleNet
        Args: x, list of tensors
        Returns: out, tensor, 4D, [B, C, H, W]
    '''
    def __init__(self, num_feature=64, out_channel=3, norm='kaiming'):
        super().__init__()
        
        self.conv_block = nn.Sequential(OrderedDict([
            ('res', ResConvBlock(num_feature)),
            ('conv', nn.Conv2d(num_feature, out_channel, 3, padding=1, bias=True)),
        ]))
        self.init_weights(norm)
        
    def init_weights(self, norm='kaiming'):
        for name, param in self.named_modules():
            if isinstance(param, nn.Conv2d) and name.find('res')==-1:
                nn.init.kaiming_normal_(param.weight)
    
    def forward(self, x):
        inputs, pyramids = x
        final_out = self.conv_block(inputs[0])
        
        return pyramids[0] - final_out


class MalleNet(nn.Module):
    def __init__(self, in_channel, num_feature, out_channel=None, low_res='down', down_scale=2, stage=3, depth=3):
        super().__init__()
        
        out_channel = in_channel if out_channel is None else out_channel     
        self.model_1 = ModelOne(in_channel=in_channel, num_feature=num_feature, down_scale=down_scale, depth=depth)
        self.model_2 = ModelTwo(num_feature=num_feature, low_res=low_res, stage=stage)
        self.model_3 = ModelThree(num_feature=num_feature, out_channel=out_channel)
        
    def forward(self, x):
        x = self.model_1(x)
        x = self.model_2(x)        
        out = self.model_3(x)        
        
        return torch.clip(out, 0.0, 1.0)
