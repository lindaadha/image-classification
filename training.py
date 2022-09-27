import torch
from torchvision import transforms, datasets
import numpy as np 
from torch.utils.data import DataLoader
from src.modules.backbone import mobilenetv2

transformations = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

batch_size = 10 
train_set = 'split_dataset/train'
train_data = datasets.ImageFolder(train_set, transform=transformations)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
print("The number of images in a training set is: ", len(train_loader)*batch_size)

test_set = 'split_dataset/test'
test_data = datasets.ImageFolder(test_set, transform=transformations)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
print("The number of images in a test set is: ", len(test_loader)*batch_size)

print("The number of batches per epoch is: ", len(train_loader))
# classes = ('bike','car','helicopter','ship','truck')
classes = train_loader.dataset.classes
print(classes)

import torch.optim as optim
from torch import nn 

model = mobilenetv2.MobileNetV2(3,(244,244),5).to('cpu') #rgb, w+h, class

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.fit(train_loader, test_loader, 100, criterion, optimizer, 'cpu')

torch.save(model,'vehicle.pth') #save model 
