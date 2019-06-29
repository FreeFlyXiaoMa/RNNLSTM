# -*- coding: utf-8 -*-
#@Time    :2019/6/27 16:21
#@Author  :XiaoMa
import torch as t

import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import numpy as np
#show=ToPILImage()

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset=tv.datasets.CIFAR10(
    root='/pytorch_demo/',
    train=True,
    download=True,
    transform=transform
)

trainloader=t.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

#测试集
testset=tv.datasets.CIFAR10(
    '/pytorch_demo/',
    train=False,
    download=True,
    transform=transform
)
testloader=t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

classes =('plane','car','bird','cat','deer','dog','frog','horse',
          'ship','truck')

(data,label)=trainset[100]
print(classes[label])

#show((data+1)/2).resize((100,100))

dataiter=iter(trainloader)
images,labels=dataiter.next()   #返回4张图片及标签

print(' .join(%11s'%classes[labels[j]] for j in range(4))
imshow(tv.utils.make_grid(images))
#print labels
print(' '.join('%5s'%classes[labels[j]] for j in range(4)))


