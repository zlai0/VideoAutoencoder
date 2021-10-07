import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from .submodule import conv, get_resnet50, stn


class Encoder3D(nn.Module):
    def __init__(self, args):
        super(Encoder3D, self).__init__()
        self.feature_extraction = get_resnet50()
        self.conv3d_1 = nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1)
        self.conv3d_2 = nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1)

    def forward(self, img):
        z_2d = self.feature_extraction(img)
        B,C,H,W = z_2d.shape
        z_3d = z_2d.reshape([-1, 256, 8, H, W])
        z_3d = F.leaky_relu(self.conv3d_1(z_3d))
        z_3d = F.leaky_relu(self.conv3d_2(z_3d))
        return z_3d

class EncoderTraj(nn.Module):
    def __init__(self, args):
        super(EncoderTraj, self).__init__()
        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(6, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.pose_pred = nn.Conv2d(conv_planes[6], 6, kernel_size=1, padding=0)

        self.scale_rotate = args.scale_rotate
        self.scale_translate = args.scale_translate

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, input):
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = pose.view(pose.size(0), 6)

        pose_r = pose[:,:3] * self.scale_rotate
        pose_t = pose[:,3:] * self.scale_translate

        pose_final = torch.cat([pose_r, pose_t], 1)

        return pose_final

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.depth_3d = 32
        self.conv3 = nn.Conv2d(2048, 512, 1)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.upconv_final = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, code):
        code = code.view(-1, code.size(1) * code.size(2), code.size(3), code.size(4))
        code = F.leaky_relu(self.conv3(code))
        code = F.leaky_relu(self.upconv3(code))
        code = F.leaky_relu(self.upconv4(code))
        output = self.upconv_final(code)
        return output


class Rotate(nn.Module):
    def __init__(self, args):
        super(Rotate, self).__init__()
        self.padding_mode = args.padding_mode
        self.conv3d_1 = nn.Conv3d(32,64,3,padding=1)
        self.conv3d_2 = nn.Conv3d(64,64,3,padding=1)

    def forward(self, code, theta):
        rot_code = stn(code, theta, self.padding_mode)
        rot_code = F.leaky_relu(self.conv3d_1(rot_code))
        rot_code = F.leaky_relu(self.conv3d_2(rot_code))
        return rot_code