import torchinfo
from VGG import *

# 搭建网络
conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))
net = vgg(conv_arch)

# 构造输入，测试模型
input = torch.rand((1,3,32,32))
torchinfo.summary(net,input.shape)

for p in net.parameters():
    print(p.shape,p.numel())
total_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total_params}")

num = 0
with open("demo.txt","r",encoding = 'utf-8') as f:
    for line in f:
        num += int(line)
print(num)