import torch 
import cv2
import numpy as np
from torchvision import transforms 
import torch.nn.functional as F

classes = ('bike','car','helicopter','ship','truck')

image = cv2.imread("ship.jpg")
image = cv2.resize(image, (224,224), interpolation=cv2.INTER_NEAREST)
MEAN = 255 * np.array([0.485, 0.456, 0.406])
STD = 255 * np.array([0.229, 0.224, 0.225])

image = image.transpose(-1,0,1)
image = (image - MEAN[:, None, None]) / STD[:, None, None]

image = image.astype(np.float32)

x = torch.from_numpy(image)
x = torch.unsqueeze(x, 0)

model = torch.load("vehicle.pth")
output = model(x)

x = torch.softmax(output,1)
x = x.detach().numpy()
print(x)
result = np.argmax(x)
print(x[0][result])
pred = classes[result]

# print(model(pred))
# print(model(result))
