import numpy as np
import pandas as pd
import matplotlib.pyplot as plt        #三剑客库
from PIL import Image              #图像处理库
import time                       #时间库
import torch                      #pytorch库
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim       #这几个都是pytorch中的网络和数据相关库
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models      #torchvision中的数据转换库、数据、模型

## ResidualBlock残差块的网络结构
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        ## channels:b表示要输入的feature map 数量
        super(ResidualBlock, self).__init__()     #类的继承
        self.conv = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),    #卷积层，内核为3*3，步长为1，填充宽度为1
            nn.ReLU(),           #Relu为激活函数
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1)     #第二个卷积层，内核为3*3，步长为1，填充宽度为1
        )

    def forward(self, x):
        return F.relu(self.conv(x) + x)    #传播

## 定义图像转换网络
class ImfwNet(nn.Module):
    def __init__(self):
        super(ImfwNet, self).__init__()   ##类的继承
        self.downsample = nn.Sequential(    ##维度扩充
            nn.ReflectionPad2d(padding=4),##使用边界反射填充
            nn.Conv2d(3,32,kernel_size=9,stride=1),    ##卷积层，3维到32维，内核9*9，步长为1
            nn.InstanceNorm2d(32,affine=True),## 在像素值上做归一化
            nn.ReLU(),  ## 3*256*256->32*256*256  激活函数Relu（）
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(32,64,kernel_size=3,stride=2),  ##卷积层，32维到64维，内核3*3，步长2
            nn.InstanceNorm2d(64,affine=True),  ##归一化
            nn.ReLU(),  ## 32*256*256 -> 64*128*128
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(64,128,kernel_size=3,stride=2),
            nn.InstanceNorm2d(128,affine=True),
            nn.ReLU(),  ## 64*128*128 -> 128*64*64
        )
        self.res_blocks = nn.Sequential(   ##块的叠加，上面定义的残差块结构
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),          ##五个残差块的堆叠
        )
        self.unsample = nn.Sequential(         ##维度缩减
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1), ##卷积层，128维到64维，内核3*3，步长2，填充宽度1。
            nn.InstanceNorm2d(64,affine=True),##归一化
            nn.ReLU(),  ## 128*64*64->64*128*128
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.InstanceNorm2d(32,affine=True),
            nn.ReLU(),  ## 64*128*128->32*256*256
            nn.ConvTranspose2d(32,3,kernel_size=9,stride=1,padding=4),## 32*256*256->3*256*256;
        )
    def forward(self,x):   ##传播
        x = self.downsample(x) ## 输入像素值－2.1～2.7之间
        x = self.res_blocks(x)
        x = self.unsample(x) ## 输出像素值－2.1～2.7之间
        return x
fwnet = ImfwNet()   ##定义网络
print(fwnet)  ##输出网络
## 数据准备
## 定义一个读取风格图像或内容图像的函数，并且将图像进行必要转化
def load_image(img_path,shape=None):
    image = Image.open(img_path).convert('RGB')  ##打开图片
    size = image.size
    ## 如果指定了图像的尺寸，就将图像转化为shape指定的尺寸
    if shape is not None:
        size = shape
    ## 使用transforms将图像转化为张量，并进行标准化
    in_transform = transforms.Compose(
        [transforms.Resize(size), # 图像尺寸变换
         transforms.ToTensor(), # 数组转化为张量
         ## 图像进行标准化
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))])
    # 使用图像的RGB通道，并且添加batch纬度
    image = in_transform(image)[:3,:,:].unsqueeze(dim=0)
    return image

# 定义一个将标准化后的图像转化为便于利用matplotlib可视化的函数
def img_convert(tensor):
    """
    将[1, c, h, w]纬度的张量转化为[ h, w,c]的数组
    因为张量进行了表转化，所以要进行标准化逆变换
    """
    image = tensor.data.numpy().squeeze() # 去处batch纬度数据
    image = image.transpose(1,2,0) ## 置换数组的纬度[c,h,w]->[h,w,c]
    ## 进行标准化的逆操作
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1) ##  将图像的取值剪切到0～1之间
    return image
# 读取内容图像
# 读取内容图像
content= load_image("meigui.png",shape = (256,256))
print("content shape:",content.shape)
## 可视化图像
plt.figure()
plt.imshow(img_convert(content))
plt.show()
#COCO风格转变
device = torch.device('cpu')   #在CPU上运行
fwnet = ImfwNet()      #定义网络
fwnet.load_state_dict(torch.load("imfwnet_dict.pkl", map_location=device))
transform_content = fwnet(content)   #图像转化
## 可视化图像
plt.figure()
plt.imshow(img_convert(transform_content))
plt.show()
#星空风格转变
device = torch.device('cpu')  #在CPU上运行
fwnet = ImfwNet()       #定义网络
fwnet.load_state_dict(torch.load("imfwnet_xingkong_dict.pkl", map_location=device))
transform_content = fwnet(content)   #图像转化
## 可视化图像
plt.figure()
plt.imshow(img_convert(transform_content))
plt.show()