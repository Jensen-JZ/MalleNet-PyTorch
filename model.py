import torch
import numpy as np
import torch.nn as nn
import bilateral_slice
from collections import OrderedDict


def space_2_depth(x, down_scale):
    B, C, H, W = x.shape
    out_channel = C * (down_scale**2)
    height = H // down_scale
    width = W // down_scale
    
    x = x.view(B*C, height, down_scale, width, down_scale)
    out = x.permute(0, 2, 4, 1, 3).contiguous().view(B, out_channel, height, width)
    
    return out


class ShufflePyramidDecom(nn.Module):
    def __init__(self, down_scale, num_high=3):
        super().__init__()
        self.down_scale = down_scale
        self.num_high = num_high
        
    def __call__(self, x):
        current = x
        pyramids = []
        pyramids.append(current)
        for i in range(self.num_high):
            current = space_2_depth(current, self.down_scale)
            pyramids.append(current)
        
        return pyramids
    

class ResConvBlock(nn.Module):
    '''Inverted Bottleneck Block
    '''
    def __init__(self, in_channel=64):
        super().__init__()
        self.convblock = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channel, in_channel*5, kernel_size=1, bias=True)), 
            ('norm1', nn.BatchNorm2d(in_channel*5)),
            ('prelu', nn.PReLU(init=0.0)),
            ('dwconv', nn.Conv2d(in_channel*5, in_channel*5, kernel_size=3, padding=1, bias=True, 
groups=in_channel*5)),
            ('norm2', nn.BatchNorm2d(in_channel*5)),
            ('prelu', nn.PReLU(init=0.0)),
            ('conv2', nn.Conv2d(in_channel*5, in_channel, kernel_size=1, bias=True)),
            ('norm3', nn.BatchNorm2d(in_channel)), 
            ('prelu', nn.PReLU(init=0.0)),
        ]))
        
    def init_weights(self):
        for name, param in self.named_modules():
            if isinstance(param, nn.Conv2d):
                nn.init.kaiming_normal_(param.weight) 
         
    def forward(self, x):
        identify = x
        self.init_weights()
        out = x + self.convblock(x)
        
        return out
    
    
class ModelOne(nn.Module):
    '''MalleNet takes a 4-level image pyramid as input.
    '''
    def __init__(self, in_channel=3, num_feature=64, down_scale=2, depth=3):
        super().__init__()
        
        self.shuffle_pyramid_decom = ShufflePyramidDecom(down_scale=down_scale, num_high=depth)
        self.pyramids = []
        self.conv_pyramid_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel, num_feature, 3, 
                                padding=1, bias=True)),
            ('res', ResConvBlock(num_feature)),
        ]))
        self.conv_pyramid_2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel*(down_scale**2), 
                                num_feature, 3, padding=1, bias=True)),
            ('res', ResConvBlock(num_feature)),
        ]))
        self.conv_pyramid_3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel*(down_scale**2)**2, 
                                num_feature*2, 3, padding=1, bias=True)),
            ('res', ResConvBlock(num_feature*2)),
        ]))
        self.conv_pyramid_4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channel*(down_scale**2)**3, 
                                num_feature*4, 3, padding=1, bias=True)),
            ('res', ResConvBlock(num_feature*4)),
        ]))
        self.pyramids += [self.conv_pyramid_1, self.conv_pyramid_2, 
                          self.conv_pyramid_3, self.conv_pyramid_4]
        
    def init_weights(self):
        for name, param in self.named_modules():
            if isinstance(param, nn.Conv2d) and name.find('res')==-1:
                nn.init.kaiming_normal_(param.weight)
        
    def forward(self, x):
        pyramid_outs = self.shuffle_pyramid_decom(x)
        self.init_weights()
        outs = []
        for i, feature in enumerate(pyramid_outs):
            outs.append(self.pyramids[i](feature))
        
        return outs, pyramid_outs
    
    
class Policy(nn.Module):
    def __init__(self, in_channel=3, num_feature=64, n_in=65, n_out=64, gz=1, stage=3):
        super().__init__()
        self.stage = stage
        self.n_in = n_in
        self.n_out = n_out
        
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
            self.res_pool_3 = nn.MaxPool2d(kernel_size=2)
            low_block_4 = [
                ResConvBlock(in_channel=num_feature)
                for i in range(3)
            ]
            self.res_block_4 = nn.Sequential(*res_block_4)
        
        self.low_dense_2 = nn.Conv2d(num_feature, gz*n_in*n_out, 1, padding=0, bias=True)
        
    def init_weights(self):
        for name, param in self.named_modules():
            if isinstance(param, nn.Conv2d) and name.find('res')==-1:
                nn.init.kaiming_normal_(param.weight)
    
    def forward(self, x):
        self.init_weights()
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
        out = torch.stack(torch.split(out, out.shape[1]//(self.n_in*self.n_out), dim=1), dim=2).permute(0, 2, 3, 
4, 1)
        
        return out
    
    
class LargePolicy(nn.Module):
    def __init__(self, in_channel=3, num_feature=64, n_in=65, n_out=64, gz=1, stage=3):
        super().__init__()
        self.stage = stage
        self.n_in = n_in
        self.n_out = n_out
        
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
                ('prelu', nn.PReLU(init=0.0)),
            ]))
            res_block_4 = [
                ResConvBlock(num_feature*8)
                for i in range(3)
            ]
            self.res_block_4 = nn.Sequential(*res_block_4)
        
        self.stage=4 if self.stage > 3 else self.stage
        self.low_dense_2 = nn.Conv2d(num_feature*2**(self.stage-1), gz*n_in*n_out, 
                                    1, padding=0, bias=True)
        
    def init_weights(self):
        for name, param in self.named_modules():
            if isinstance(param, nn.Conv2d) and name.find('res')==-1:
                nn.init.kaiming_normal_(param.weight)
    
    
    def forward(self, x):
        self.init_weights()
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
        out = torch.stack(torch.split(out, out.shape[1]//(self.n_in*self.n_out), dim=1), dim=2).permute(0, 2, 3, 
4, 1)
        
        return out
    
    
class BilateralSliceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bilateral_grid, guide, input, has_offset):
        ctx.save_for_backward(bilateral_grid, guide, input)
        ctx.has_offset = has_offset
        return bilateral_slice.forward(bilateral_grid, guide, input, has_offset)

    @staticmethod
    def backward(ctx, grad):
        bilateral_grid, guide, input = ctx.saved_variables
        d_grid, d_guide, d_input = bilateral_slice.backward(grad,
                                                            bilateral_grid,
                                                            guide,
                                                            input,
                                                            ctx.has_offset)
        return d_grid, d_guide, d_input, None
    

class ModelTwo(nn.Module):
    def __init__(self, num_features=64, gz=1, low_res='down', stage=3, group=64):
        super().__init__()
        
        self.stage = stage
        self.n_in = 1
        self.n_out = 2
        self.gz = gz
        self.group = group
        self.bilateral_slice = BilateralSliceFunction()
        
        if low_res == 'down':
            low_res_blocks = [
                Policy(in_channels=num_features*2**(i-1 if i-1>0 else 0), num_features=num_features, 
                       n_in=self.n_in, n_out=self.n_out, gz=(self.gz*2**(i-1 if i-1>0 else 0))*self.group, 
                       stage=self.stage) for i in range(4)
            ]
            self.low_res_blocks = nn.Sequential(*low_res_blocks)
            
        elif low_res == 'downavg2':
            low_res_conv = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(kernel_size=2, stride=2)), 
                ('policy', Policy(in_channels=num_features, num_features=num_features, 
                                  n_in=self.n_in, n_out=self.n_out, gz=self.gz*self.group, 
                                  stage=self.stage)),
            ]))
            low_res_blocks = [low_res_conv for i in range(2)]
            low_res_blocks.append(Policy(in_channels=num_features*2, num_features=num_features, 
                                         n_in=self.n_in, n_out=self.n_out, gz=self.gz*self.group*2, 
                                         stage=self.stage))
            low_res_blocks.append(Policy(in_channels=num_features*4, num_features=num_features, 
                                         n_in=self.n_in, n_out=self.n_out, gz=self.gz*self.group*4, 
                                         stage=self.stage))
            
            self.low_res_blocks = nn.Sequential(*low_res_blocks)
            
        elif low_res == 'ldownavg2':
            low_res_conv = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(kernel_size=2, stride=2)), 
                ('largepolicy', LargePolicy(in_channels=num_features, num_features=num_features, 
                                            n_in=self.n_in, n_out=self.n_out, gz=self.gz*self.group, 
                                            stage=self.stage)),
            ]))
            low_res_blocks = [low_res_conv for i in range(2)]
            low_res_blocks.append(LargePolicy(in_channels=num_features*2, num_features=num_features, 
                                              n_in=self.n_in, n_out=self.n_out, gz=self.gz*self.group*2, 
                                              stage=self.stage))
            low_res_blocks.append(LargePolicy(in_channels=num_features*4, num_features=num_features, 
                                              n_in=self.n_in, n_out=self.n_out, gz=self.gz*self.group*4, 
                                              stage=self.stage))
            
            self.low_res_blocks = nn.Sequential(*low_res_blocks)
            
        elif low_res == 'downavg4':
            low_res_conv = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(kernel_size=4, stride=4)), 
                ('policy', Policy(in_channels=num_features, num_features=num_features, 
                                  n_in=self.n_in, n_out=self.n_out, gz=self.gz*self.group, 
                                  stage=self.stage)),
            ]))
            low_res_blocks = [low_res_conv for i in range(2)]
            low_res_blocks.append(Policy(in_channels=num_features*2, num_features=num_features, 
                                         n_in=self.n_in, n_out=self.n_out, gz=self.gz*self.group*2, 
                                         stage=self.stage))
            low_res_blocks.append(Policy(in_channels=num_features*4, num_features=num_features, 
                                         n_in=self.n_in, n_out=self.n_out, gz=self.gz*self.group*4, 
                                         stage=self.stage))
            
            self.low_res_blocks = nn.Sequential(*low_res_blocks)
            
        elif low_res == 'ldownavg4':
            low_res_conv = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(kernel_size=4, stride=4)), 
                ('largepolicy', LargePolicy(in_channels=num_features, num_features=num_features, 
                                            n_in=self.n_in, n_out=self.n_out, gz=self.gz*self.group, 
                                            stage=self.stage)),
            ]))
            low_res_blocks = [low_res_conv for i in range(2)]
            low_res_blocks.append(LargePolicy(in_channels=num_features*2, num_features=num_features, 
                                              n_in=self.n_in, n_out=self.n_out, gz=self.gz*self.group*2, 
                                              stage=self.stage))
            low_res_blocks.append(LargePolicy(in_channels=num_features*4, num_features=num_features, 
                                              n_in=self.n_in, n_out=self.n_out, gz=self.gz*self.group*4, 
                                              stage=self.stage))
            
            self.low_res_blocks = nn.Sequential(*low_res_blocks)
            
        elif low_res == 'unet':
            low_res_blocks = [nn.Identity() for i in range(4)]
            self.low_res_blocks = nn.Sequential(*low_res_blocks)  # has some bugs.
        
        bilateral_blocks = [
            nn.Sequential(OrderedDict([
                ('norm', nn.BatchNorm2d(64*2**(i-1 if i-1>0 else 0))), 
                ('prelu', nn.PReLU(init=0.0)),
            ])) for i in range(4)
        ]
        self.bilateral_blocks = nn.Sequential(*bilateral_blocks)
        self.gate = [torch.tensor([0.], requires_grad=True) for i in range(4)]
        
        self.conv_4_1 = nn.Sequential(OrderedDict([
            ('res1', ResConvBlock(num_features*4)),
            ('res2', ResConvBlock(num_features*4)),
        ]))
        self.conv_4_2 = nn.Sequential(OrderedDict([
            ('res1', ResConvBlock(num_features*4)),
            ('res2', ResConvBlock(num_features*4)),
        ]))
        self.conv_3_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(num_features*6, num_features*2, 3, padding=1, bias=True)),
            ('norm', nn.BatchNorm2d(num_features*2)),
            ('res1', ResConvBlock(num_features*2)),
            ('res2', ResConvBlock(num_features*2)),
        ]))
        self.conv_3_2 = nn.Sequential(OrderedDict([
            ('res1', ResConvBlock(num_features*2)),
            ('res2', ResConvBlock(num_features*2)),
        ]))
        self.conv_2_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(num_features*3, num_features, 3, padding=1, bias=True)),
            ('norm', nn.BatchNorm2d(num_features)),
            ('res1', ResConvBlock(num_features)),
            ('res2', ResConvBlock(num_features)),
        ]))
        self.conv_2_2 = nn.Sequential(OrderedDict([
            ('res1', ResConvBlock(num_features)),
            ('res2', ResConvBlock(num_features)),
        ]))
        self.conv_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(num_features*2, num_features, 1, padding=0, bias=True)),
            ('norm', nn.BatchNorm2d(num_features)),
        ]))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def init_weights(self):
        for name, param in self.named_modules():
            if isinstance(param, nn.Conv2d) and name.find('res')==-1:
                nn.init.kaiming_normal_(param.weight)

    def forward(self, x):
        self.init_weights()
        inputs, pyramids = x
        outs = []
        image_1, image_2, image_3, image_4 = inputs
        # level_4
        image_4 = self.conv_4_1(image_4)
        # ************** MalleConv ****************
        group = self.group * 4
        content_feature = image_4
        # ********* Efficient Predictor Network ***********
        grid_coefficients = self.low_res_blocks[-1](content_feature)
        grid_coefficients = torch.stack(torch.split(grid_coefficients, grid_coefficients.shape[4]//group, dim=4), 
dim=5)
        # ********* Efficient Predictor Network ***********
        post_image = []
        # ********* Efficient On-the-fly Slicing ***********
        for j in range(group):
            post_image_j = self.bilateral_slice.apply(  # args: grid (N, C, H, W, D), guide (N, H, W), input (N, 
C, H, W);
                grid_coefficients[:, :, :, :, :, j].contiguous(),
                torch.clip(content_feature[:, j, :, :], 0, 1),
                content_feature[:, j:j+1, :, :].contiguous(),
                True
            )
            post_image.append(post_image_j)
        post_image = torch.cat(post_image, dim=1)
        post_image = self.bilateral_blocks[3](post_image)
        post_image = torch.clip(post_image, -1, 1)
        image_4 = image_4 + post_image * self.gate[3].to(post_image.device)
        # ********* Efficient On-the-fly Slicing ***********
        image_4 = self.conv_4_2(image_4)
        # ************** MalleConv ****************
        
        # level_3
        image_3 = self.conv_3_1(torch.cat([image_3, self.upsample(image_4)], dim=1))
        group = self.group * 2
        content_feature = image_3
        grid_coefficients = self.low_res_blocks[-2](content_feature)
        grid_coefficients = torch.stack(torch.split(grid_coefficients, grid_coefficients.shape[4]//group, dim=4), 
dim=5)
        post_image = []
        for j in range(group):
            post_image_j = self.bilateral_slice.apply(
                grid_coefficients[:, :, :, :, :, j].contiguous(),
                torch.clip(content_feature[:, j, :, :], 0, 1),
                content_feature[:, j:j+1, :, :].contiguous(),
                True
            )
            post_image.append(post_image_j)
        post_image = torch.cat(post_image, dim=1)
        post_image = self.bilateral_blocks[2](post_image)
        post_image = torch.clip(post_image, -1, 1)
        image_3 = image_3 + post_image * self.gate[2].to(post_image.device)
        image_3 = self.conv_3_2(image_3)
        
        # level_2
        image_2 = self.conv_2_1(torch.cat([image_2, self.upsample(image_3)], dim=1))
        group = self.group
        content_feature = image_2
        grid_coefficients = self.low_res_blocks[-3](content_feature)
        grid_coefficients = torch.stack(torch.split(grid_coefficients, grid_coefficients.shape[4]//group, dim=4), 
dim=5)
        post_image = []
        for j in range(group):
            post_image_j = self.bilateral_slice.apply(
                grid_coefficients[:, :, :, :, :, j].contiguous(),
                torch.clip(content_feature[:, j, :, :], 0, 1),
                content_feature[:, j:j+1, :, :].contiguous(),
                True
            )
            post_image.append(post_image_j)
        post_image = torch.cat(post_image, dim=1)
        post_image = self.bilateral_blocks[1](post_image)
        post_image = torch.clip(post_image, -1, 1)
        image_2 = image_2 + post_image * self.gate[1].to(post_image.device)
        image_2 = self.conv_2_2(image_2)
        
        # level_1
        image_1 = self.conv_1(torch.cat([image_1, self.upsample(image_2)], dim=1))
        group = self.group
        content_feature = image_1
        grid_coefficients = self.low_res_blocks[0](content_feature)
        grid_coefficients = torch.stack(torch.split(grid_coefficients, grid_coefficients.shape[4]//group, dim=4), 
dim=5)
        print(grid_coefficients.shape)
        post_image = []
        for j in range(group):
            post_image_j = self.bilateral_slice.apply(
                grid_coefficients[:, :, :, :, :, j].contiguous(),
                torch.clip(content_feature[:, j, :, :], 0, 1),
                content_feature[:, j:j+1, :, :].contiguous(),
                True
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
    def __init__(self, num_feature=64, out_channel=3):
        super().__init__()
        
        self.convblock = nn.Sequential(OrderedDict([
            ('res', ResConvBlock(num_feature)),
            ('conv', nn.Conv2d(num_feature, out_channel, 3, padding=1, bias=True)),
        ]))
        
    def init_weights(self):
        for name, param in self.named_modules():
            if isinstance(param, nn.Conv2d) and name.find('res')==-1:
                nn.init.kaiming_normal_(param.weight)
        
    def forward(self, x):
        self.init_weights()
        inputs, pyramids = x
        outs = []
        final_out = self.convblock(inputs[0])
        return pyramids[0] - final_out
    
    
class MalleNet(nn.Module):
    def __init__(self, in_channel=3, num_feature=64, out_channel=3, low_res="down", down_scale=2, stage=3, 
depth=3, gz=1, n_in=1, n_out=2, group=64):
        super().__init__()
        
        self.model_1 = ModelOne(in_channel=in_channel, num_feature=num_feature, down_scale=down_scale, 
depth=depth)
        self.model_2 = ModelTwo(num_feature=num_feature, low_res=low_res, gz=gz, stage=stage, group=group)
        self.model_3 = ModelThree(num_feature=num_feature, out_channel=out_channel)
    
    def forward(self, x):
        x = self.model_1(x)
        x = self.model_2(x)
        x = self.model_3(x)
        
        return x
        
