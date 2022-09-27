import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import glob

test_dir = 'split_dataset/cek/'
classes = ('bike','car','helicopter','ship','truck')
# model = torch.load("vehicle.pth")


def prediction(img_dir):
    image = cv2.imread(img_dir)
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

    x = torch.softmax(output, 1)
    x = x.detach().numpy()
    result = np.argmax(x)
    prob = x[0][result]
    pred = classes[result]

    return pred, prob

for img in os.listdir(test_dir):
    img_name = os.path.join(test_dir, img)
    predict, prob = prediction(img_name)

    print(img + "=" + predict)
    print("Prob: ", prob*100)

    # print(img)


