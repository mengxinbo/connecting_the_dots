import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.networks import TimedModule, DispNetS, OutputLayerFactory, DispNetShallow


class PolarizationDispEdgeDecoders(TimedModule):
    '''
    Polarization Camera Disparity and Edge Decoders
    
    Automatically adjusts input channels based on polarization input type.
    '''
    def __init__(self, channels_in, imsizes, max_disp=128, output_ms=True, coordconv=False, weight_init=False, channel_multiplier=1):
        super(PolarizationDispEdgeDecoders, self).__init__(mod_name='PolarizationDispEdgeDecoders')
        
        # Disparity decoder
        output_facs = [OutputLayerFactory(type='disp', params={'alpha': max_disp/(2**s), 'beta': 0, 'gamma': 1, 'offset': 3}) for s in range(4)]
        self.disp_decoder = DispNetS(channels_in, imsizes, output_facs=output_facs, output_ms=output_ms, coordconv=coordconv, weight_init=weight_init, channel_multiplier=channel_multiplier)
        
        # Edge decoder
        output_facs = [OutputLayerFactory(type='linear') for s in range(4)]
        self.edge_decoder = DispNetShallow(channels_in, imsizes, output_facs=output_facs, output_ms=output_ms, coordconv=coordconv, weight_init=weight_init)
    
    def tforward(self, x):
        disp = self.disp_decoder(x)
        edge = self.edge_decoder(x)
        return disp, edge


class PolarizationNormalDecoder(TimedModule):
    '''
    Surface Normal Decoder for Polarization Camera
    
    Estimates surface normals from polarization information.
    This leverages the unique advantage of polarization cameras.
    '''
    def __init__(self, channels_in, imsizes, output_ms=True, coordconv=False, weight_init=False, channel_multiplier=1):
        super(PolarizationNormalDecoder, self).__init__(mod_name='PolarizationNormalDecoder')
        
        self.output_ms = output_ms
        self.coordconv = coordconv
        
        conv_planes = channel_multiplier * np.array([32, 64, 128, 256, 512, 512, 512])
        self.conv1 = self.downsample_conv(channels_in, conv_planes[0], kernel_size=7)
        self.conv2 = self.downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = self.downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = self.downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = self.downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = self.downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = self.downsample_conv(conv_planes[5], conv_planes[6])
        
        upconv_planes = channel_multiplier * np.array([512, 512, 256, 128, 64, 32, 16])
        self.upconv7 = self.upconv(conv_planes[6], upconv_planes[0])
        self.upconv6 = self.upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = self.upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = self.upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = self.upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = self.upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = self.upconv(upconv_planes[5], upconv_planes[6])
        
        self.iconv7 = self.conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = self.conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = self.conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = self.conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = self.conv(3 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = self.conv(3 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = self.conv(3 + upconv_planes[6], upconv_planes[6])
        
        # Normal prediction layers - output 3 channels for (nx, ny, nz)
        self.predict_normal4 = nn.Conv2d(upconv_planes[3], 3, kernel_size=3, padding=1)
        self.predict_normal3 = nn.Conv2d(upconv_planes[4], 3, kernel_size=3, padding=1)
        self.predict_normal2 = nn.Conv2d(upconv_planes[5], 3, kernel_size=3, padding=1)
        self.predict_normal1 = nn.Conv2d(upconv_planes[6], 3, kernel_size=3, padding=1)
    
    def downsample_conv(self, in_planes, out_planes, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(inplace=True)
        )
    
    def conv(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
    
    def crop_like(self, input, ref):
        assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
        return input[:, :, :ref.size(2), :ref.size(3)]
    
    def normalize_normal(self, normal):
        """Normalize normal vectors to unit length"""
        norm = torch.sqrt(torch.sum(normal**2, dim=1, keepdim=True) + 1e-8)
        return normal / norm
    
    def tforward(self, x):
        # Encoder
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        
        # Decoder
        out_upconv7 = self.crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        
        out_upconv6 = self.crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        
        out_upconv5 = self.crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        
        out_upconv4 = self.crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        normal4 = self.normalize_normal(self.predict_normal4(out_iconv4))
        
        out_upconv3 = self.crop_like(self.upconv3(out_iconv4), out_conv2)
        normal4_up = self.crop_like(F.interpolate(normal4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, normal4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        normal3 = self.normalize_normal(self.predict_normal3(out_iconv3))
        
        out_upconv2 = self.crop_like(self.upconv2(out_iconv3), out_conv1)
        normal3_up = self.crop_like(F.interpolate(normal3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, normal3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        normal2 = self.normalize_normal(self.predict_normal2(out_iconv2))
        
        out_upconv1 = self.crop_like(self.upconv1(out_iconv2), x)
        normal2_up = self.crop_like(F.interpolate(normal2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, normal2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        normal1 = self.normalize_normal(self.predict_normal1(out_iconv1))
        
        if self.output_ms:
            return normal1, normal2, normal3, normal4
        else:
            return normal1


class PolarizationMultiTaskNet(TimedModule):
    '''
    Multi-task Network for Polarization Camera
    
    Jointly estimates:
    - Disparity/Depth
    - Edge
    - Surface Normal (leveraging polarization information)
    '''
    def __init__(self, channels_in, imsizes, max_disp=128, output_ms=True, coordconv=False, weight_init=False, channel_multiplier=1):
        super(PolarizationMultiTaskNet, self).__init__(mod_name='PolarizationMultiTaskNet')
        
        self.disp_edge_decoder = PolarizationDispEdgeDecoders(
            channels_in, imsizes, max_disp=max_disp, 
            output_ms=output_ms, coordconv=coordconv, 
            weight_init=weight_init, channel_multiplier=channel_multiplier
        )
        
        self.normal_decoder = PolarizationNormalDecoder(
            channels_in, imsizes, 
            output_ms=output_ms, coordconv=coordconv, 
            weight_init=weight_init, channel_multiplier=channel_multiplier
        )
    
    def tforward(self, x):
        disp, edge = self.disp_edge_decoder(x)
        normal = self.normal_decoder(x)
        return disp, edge, normal


if __name__ == '__main__':
    pass
