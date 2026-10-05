from turtle import forward
from torch._dynamo.convert_frame import input_codes
from utils.dataset import get_loader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim
from tqdm import tqdm

class MSFEM(nn.Module):
    def __init__(self, input_size = 64, output_size = 64) -> None:
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels= input_size, out_channels= input_size, kernel_size = 1, stride = 1, padding = 0)
        self.branch2 = nn.Conv2d(in_channels= input_size, out_channels= input_size, kernel_size = 3, stride = 1, padding = 1)
        self.branch3 = nn.Conv2d(in_channels= input_size, out_channels= input_size, kernel_size = 5, stride = 1, padding = 2)
        self.Relu = nn.ReLU(inplace = True)
        self.comb = nn.Conv2d(in_channels = 3 * input_size, out_channels = output_size, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        x1 = self.Relu(self.branch1(x))
        x2 = self.Relu(self.branch2(x))
        x3 = self.Relu(self.branch3(x))
        combination = torch.cat([x1,x2,x3], dim = 1)
        x4 = self.comb(combination)
        return x4
            
class IlluminationAttention(nn.Module):
    def __init__(self, input_size = 64) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, 1),
            nn.ReLU(True),
            
            nn.Conv2d(input_size // 2, input_size, 3, 1, 1),
            nn.Sigmoid()
        )    

    def forward(self, x):
        attn_map = self.attn(x)
        out = x * attn_map
        return x + out
        
        
class Encoder(nn.Module):
    def __init__(self, input_size = 3, output_size = 32, kernel_size = 3, stride = 1, padding = 1, use_maxpool = True, use_module = True) -> None:
        super().__init__()

        arr_layer = []
        if use_maxpool:
            arr_layer.append(nn.MaxPool2d(2))
        arr_layer.append(nn.Conv2d(in_channels=input_size, out_channels= output_size, kernel_size= kernel_size, stride = stride, padding = padding))
        arr_layer.append(nn.LeakyReLU())
        arr_layer.append(nn.BatchNorm2d(output_size))
        arr_layer.append(nn.Conv2d(in_channels=output_size, out_channels= output_size, kernel_size= kernel_size, stride = stride, padding = padding))
        arr_layer.append(nn.LeakyReLU())
        arr_layer.append(nn.BatchNorm2d(output_size))
        
        if use_module:
            arr_layer.append(MSFEM(output_size, output_size))
            arr_layer.append(IlluminationAttention(output_size))
        
        self.block = nn.Sequential(*arr_layer)
    
    def forward(self, x):
        return self.block(x)
      
class Decoder(nn.Module):
    def __init__(self, input_size = 3, output_size = 32, kernel_size = 3, stride = 1, padding = 1, use_deconv = True, use_tanh = True, use_conv = True) -> None:
        super().__init__()
        arr_layer = []
        input_deconv = input_size
        if use_tanh == True:
            output_tanh = output_size
            output_size = input_size // 4
        if use_conv:
            arr_layer.append(nn.Conv2d(in_channels=input_size, out_channels= input_size // 2, kernel_size= kernel_size, stride = stride, padding = padding))
            arr_layer.append(nn.LeakyReLU())
            arr_layer.append(nn.BatchNorm2d(input_size // 2))
            arr_layer.append(nn.Conv2d(in_channels=input_size // 2, out_channels= output_size, kernel_size= kernel_size, stride = stride, padding = padding))
            arr_layer.append(nn.LeakyReLU())
            arr_layer.append(nn.BatchNorm2d(output_size))
            input_deconv = output_size
        if use_deconv:
            arr_layer.append(nn.ConvTranspose2d(input_deconv, output_size,3,2,1))
        if use_tanh:
            arr_layer.append(nn.Conv2d(input_deconv, output_tanh, kernel_size= kernel_size, stride = stride, padding = padding))
            arr_layer.append(nn.Tanh())
        self.block = nn.Sequential(*arr_layer)
        
    def forward(self, x, skip_connection = None):
        res = self.block(x)
        
        if skip_connection is not None:
            if res.shape != skip_connection.shape:
                res = F.interpolate(res, skip_connection.shape[2:])
            res = torch.concat([res, skip_connection], dim = 1)
        
        return res

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = Encoder(3,32, use_maxpool=False)
        self.encoder2 = Encoder(32,64)
        self.encoder3 = Encoder(64,128)
        self.encoder4 = Encoder(128,256)
        self.encoder5 = Encoder(256,512, use_module=False)
        self.bottemneck = Decoder(512,256, use_conv=False, use_tanh=False)
        self.decoder4 = Decoder(512,128, use_tanh=False)
        self.decoder3 = Decoder(256,64, use_tanh=False)
        self.decoder2 = Decoder(128,32, use_tanh= False)
        self.decoder1 = Decoder(64,3, use_deconv=False, use_conv=False)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        btn = self.bottemneck(e5, e4)
        d4 = self.decoder4(btn, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)
        
        return d1
        
class Local_Discriminator(nn.Module):
        def __init__(self, in_channels = 3, kernel_size = 4, stride = 2, padding = 1) -> None:
            super().__init__()
            self.encode1 = nn.Conv2d(in_channels=in_channels, out_channels= 32, kernel_size=kernel_size, stride=stride,padding=padding)
            self.leakyRelu = nn.LeakyReLU()
            self.encode2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=kernel_size, stride=stride,padding=padding)
            self.encode3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size=kernel_size, stride=stride,padding=padding)
            self.encode4 = nn.Conv2d(in_channels=128, out_channels= 256, kernel_size=kernel_size, stride=stride,padding=padding)
            self.encode5 = nn.Conv2d(in_channels=256, out_channels= 512, kernel_size=kernel_size, stride=stride,padding=padding)
            self.encode6 = nn.Conv2d(in_channels=512, out_channels= 1, kernel_size=kernel_size, stride=1,padding=padding)

        def forward(self, x):
            e1 = self.leakyRelu(self.encode1(x))
            e2 = self.leakyRelu(self.encode2(e1))
            e3 = self.leakyRelu(self.encode3(e2))
            e4 = self.leakyRelu(self.encode4(e3))
            e5 = self.leakyRelu(self.encode5(e4))
            e6 = self.leakyRelu(self.encode6(e5))
            return e6

class Global_Discriminator(nn.Module):
    def __init__(self, in_channels = 3, kernel_size = 4, stride = 2, padding = 1) -> None:
        super().__init__()
        self.encode1 = nn.Conv2d(in_channels=in_channels, out_channels= 32, kernel_size=kernel_size, stride=stride,padding=padding)
        self.leakyRelu = nn.LeakyReLU()
        self.encode2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=kernel_size, stride=stride,padding=padding)
        
        self.dilated1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=padding)
        self.dilated2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=2, dilation=2)
        self.dilated3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=3, dilation=3)
        
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=4, padding=0)
        self.encode3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size=kernel_size, stride=stride,padding=padding)  
        self.encode4 = nn.Conv2d(in_channels=128, out_channels= 256, kernel_size=kernel_size, stride=stride,padding=padding)  
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=2, padding=0)      

        self.encode5 = nn.Conv2d(in_channels=448, out_channels= 512, kernel_size=kernel_size, stride=stride,padding=padding)  
        self.encodefix = nn.Conv2d(in_channels=512, out_channels= 512, kernel_size=kernel_size, stride=stride,padding=padding) 
        self.encode6 = nn.Conv2d(in_channels=512, out_channels= 1, kernel_size=kernel_size, stride=1,padding=1)  
        
    def forward(self, x):
        e1 = self.leakyRelu(self.encode1(x))
        e2 = self.leakyRelu(self.encode2(e1))
        
        di1 = self.leakyRelu(self.dilated1(e2))
        di2 = self.leakyRelu(self.dilated2(e2))
        di3 = self.leakyRelu(self.dilated3(e2))
        
        di_element = di1 + di2 + di3
        
        e3 = self.leakyRelu(self.encode3(di_element))
        c1 = self.conv1(di_element)
        e4 = self.leakyRelu(self.encode4(e3))
        c2 = self.conv2(e3)
        
        concat = torch.concat([c1,e4,c2], dim = 1)
        
        e5 = self.leakyRelu(self.encode5(concat))
        efix = self.leakyRelu(self.encodefix(e5))
        
        e6 = self.leakyRelu(self.encode6(efix))
        
        return e6
        
class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.global_dis = Global_Discriminator()
        self.local_dis = Local_Discriminator()
        
    def RandomCrop(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        H, W = x.shape[2:]
        crop_size = 128
        x_rand = torch.randint(0, H - crop_size + 1, (1,)).item()
        y_rand = torch.randint(0, W - crop_size + 1, (1,)).item()
        cropped_patch = x[:, :, x_rand : x_rand + crop_size, y_rand : y_rand + crop_size]
        return cropped_patch.clone().detach()
        
    def forward(self, x):
        loc = self.local_dis(self.RandomCrop(x))
        glo = self.global_dis(x)
        return loc + glo # Mục tiêu cho WxH là 3x3 
        
            
